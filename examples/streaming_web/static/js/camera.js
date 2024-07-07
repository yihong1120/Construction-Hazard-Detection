$(document).ready(function(){
    function updateImage() {
        var src = $("#camera-image").attr("src").split('?')[0]; // Remove any existing query string
        $("#camera-image").attr("src", src + '?' + new Date().getTime());
    }
    setInterval(updateImage, 5000);  // Update every 5 seconds
});
