// JavaScript Document

/***
    Utility functions available across the site
***/
var LAGUNITA = {
  size: function(){
    // return window size based on visibility as calculated by CSS
    var size = 'lg';
  
    // if we don't already have the size-detect div's, add them
    if ( $('.size-detect-xs').length == 0 ) {
      $('body')
        .append('<div class="size-detect-xs" />')
        .append('<div class="size-detect-sm" />')
        .append('<div class="size-detect-md" />')
        .append('<div class="size-detect-lg" />');
    }
  
    $(['xs', 'sm', 'md', 'lg']).each(function(i, sz) {
      if ($('.size-detect-'+sz).css('display') != 'none') {
        size = sz;
      }
    });
    return size;
  },

  stickFooter: function(){
    // adjust css to make footer sticky
    var h = $('#footer').height() + 'px';
    $('#su-content').css('padding-bottom', h);
    $('#footer').css('margin-top', '-'+h);
  }
};

$(document).ready(function() {
  LAGUNITA.stickFooter();

  // note resize events and trigger resizeEnd event when resizing stops
  $(window).resize(function() {
    if(this.resizeTO) clearTimeout(this.resizeTO);
    this.resizeTO = setTimeout(function() {
      $(this).trigger('resizeEnd');
    }, 200);
  });
  // Call responsive funtion when browser window resizing is complete
  $(window).bind('resizeEnd', function() {
    // show or hide the hamburger based on window size
    var size = LAGUNITA.size(); // what size is our window (xs, sm, md or lg)
    if (size == 'md' || size == 'lg') { // if size is md or lg, unhide search and gateway blocks
      $('.navbar-collapse').collapse('hide');
      // $('.navbar-collapse').removeClass('in').addClass('collapse');
    }
    // re-stick the footer in case its height has changed
    LAGUNITA.stickFooter();
  });
  
  $('.navbar-collapse').collapse({toggle: false}); // activate collapsibility without toggling state
  
  $('#skip > a').click(function(e){
    var href = $(this).attr('href').substr(1); // remove the #
    var target = $('a[name="' + href + '"]');
    target.focus();
  });

});
