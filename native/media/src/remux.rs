//! Reference-counted encoded packet handoff; no second input connection.
use super::*;
unsafe extern "C" {
    fn cm_decoder_packet_tap(
        pointer: *mut c_void,
        tap: extern "C" fn(*mut c_void, *mut c_void),
        opaque: *mut c_void,
    );
    fn cm_packet_retain(
        pointer: *mut c_void,
        w: *mut i32,
        h: *mut i32,
        fps: *mut f64,
        key: *mut i32,
    ) -> *mut c_void;
    fn cm_packet_free(pointer: *mut c_void);
    fn cm_remux_open(
        packet: *mut c_void,
        target: *const c_char,
        interrupt: extern "C" fn(*mut c_void) -> c_int,
        opaque: *mut c_void,
    ) -> *mut c_void;
    fn cm_remux_write(pointer: *mut c_void, packet: *mut c_void) -> c_int;
    fn cm_remux_free(pointer: *mut c_void);
}
pub struct EncodedPacket {
    pointer: NonNull<c_void>,
    pub width: i32,
    pub height: i32,
    pub fps: f64,
    pub key: bool,
}
unsafe impl Send for EncodedPacket {}
impl Drop for EncodedPacket {
    fn drop(&mut self) {
        unsafe { cm_packet_free(self.pointer.as_ptr()) }
    }
}
pub(crate) struct PacketTap {
    enabled: Box<dyn Fn() -> bool + Send + Sync>,
    send: Box<dyn Fn(Option<EncodedPacket>) + Send + Sync>,
}
extern "C" fn tap(opaque: *mut c_void, decoder: *mut c_void) {
    // No panic may unwind over C. Packet ownership is released on every path.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let tap = unsafe { &*opaque.cast::<PacketTap>() };
        if !(tap.enabled)() {
            return;
        }
        let (mut w, mut h, mut fps, mut key) = (0, 0, 0., 0);
        if let Some(pointer) =
            NonNull::new(unsafe { cm_packet_retain(decoder, &mut w, &mut h, &mut fps, &mut key) })
        {
            (tap.send)(Some(EncodedPacket {
                pointer,
                width: w,
                height: h,
                fps,
                key: key != 0,
            }));
        } else {
            (tap.send)(None);
        }
    }));
}
impl Decoder {
    pub fn tap_packets(
        &mut self,
        enabled: impl Fn() -> bool + Send + Sync + 'static,
        send: impl Fn(Option<EncodedPacket>) + Send + Sync + 'static,
    ) {
        let mut owner = Box::new(PacketTap {
            enabled: Box::new(enabled),
            send: Box::new(send),
        });
        unsafe {
            cm_decoder_packet_tap(
                self.pointer.as_ptr(),
                tap,
                (&mut *owner as *mut PacketTap).cast(),
            )
        };
        self.tap = Some(owner);
    }
}
pub struct Remuxer {
    pointer: NonNull<c_void>,
    interrupt: Box<Interrupt>,
}
unsafe impl Send for Remuxer {}
impl Remuxer {
    pub fn open(packet: &EncodedPacket, target: &str) -> Result<Self> {
        ensure!(
            target.starts_with("rtsp://") || target.starts_with("rtsps://"),
            "Passthrough requires an RTSP destination"
        );
        let target = CString::new(target)?;
        let mut interrupt = Interrupt::new(CancellationToken::new(), Duration::from_secs(5));
        let pointer = NonNull::new(unsafe {
            cm_remux_open(
                packet.pointer.as_ptr(),
                target.as_ptr(),
                interrupted,
                interrupt.pointer(),
            )
        })
        .ok_or_else(error)?;
        Ok(Self { pointer, interrupt })
    }
    pub fn write(&mut self, packet: &EncodedPacket) -> Result<()> {
        self.interrupt.reset();
        if unsafe { cm_remux_write(self.pointer.as_ptr(), packet.pointer.as_ptr()) } < 0 {
            return Err(error());
        }
        Ok(())
    }
}
impl Drop for Remuxer {
    fn drop(&mut self) {
        self.interrupt.reset();
        unsafe { cm_remux_free(self.pointer.as_ptr()) }
    }
}
