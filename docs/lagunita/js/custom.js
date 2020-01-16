// Custom Javascript for Lagunita HTML Theme
// You can add custom JS functions to this file.
jQuery(document).ready(function($){
  // Add "active" to nav element if matches body class of "subnav-n"
  var bodyClass="";
  var matches = document.body.className.match(/(^|\s+)(subnav-\d+)(\s|$)/);
  if (matches) {
    // found the bodyClass
    bodyClass = matches[2];
  }

  if (bodyClass.length > 0){
    var target = $( '#' + bodyClass );
    target.addClass('active');
  }
});