$(document).ready(function(){
    setInterval(function(){
        $('img').each(function(){
            var src = $(this).attr('src').split('?')[0]; // Remove any existing query string
            $(this).attr('src', src + '?' + new Date().getTime());
        });
    }, 5000);  // Update every 5 seconds
});
