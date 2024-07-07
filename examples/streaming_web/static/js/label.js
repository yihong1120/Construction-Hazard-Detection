$(document).ready(() => {
    // 自动检测当前页面协议，以决定 ws 还是 wss
    const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
    // 创建 WebSocket 连接，并配置重连策略
    const socket = io.connect(protocol + document.domain + ':' + location.port, {
        transports: ['websocket'],
        reconnectionAttempts: 5,   // 最多重连尝试 5 次
        reconnectionDelay: 2000    // 重连间隔为 2000 毫秒
    });

    // 获取当前页面的标签名
    const currentPageLabel = $('h1').text();  // 假设页面的 <h1> 标签包含了当前的标签名称

    socket.on('connect', () => {
        console.log('WebSocket connected!');
    });

    socket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error);
    });

    socket.on('reconnect_attempt', () => {
        console.log('Attempting to reconnect...');
    });

    socket.on('update', (data) => {
        // 检查接收到的数据是否适用于当前页面的标签
        if (data.label === currentPageLabel) {
            console.log('Received update for current label:', data.label);
            const fragment = document.createDocumentFragment();
            data.images.forEach((image, index) => {
                const cameraDiv = $('<div>').addClass('camera');
                const title = $('<h2>').text(data.image_names[index]);
                const img = $('<img>').attr('src', `data:image/png;base64,${image}`).attr('alt', `${data.label} image`);
                cameraDiv.append(title).append(img);
                fragment.appendChild(cameraDiv[0]);
            });
            $('.camera-grid').empty().append(fragment);
        } else {
            console.log('Received update for different label:', data.label);
        }
    });
});
