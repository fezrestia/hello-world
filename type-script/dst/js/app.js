var hoge;
(function (hoge) {
    class foo {
        // Get jQuery object.
        constructor(element) {
            this.element = element;
        }
        color(color) {
            this.element.css('color', color);
        }
    }
    hoge.foo = foo;
})(hoge || (hoge = {}));
$(function () {
    var foo = new hoge.foo($('div'));
    foo.color('blue');
});
