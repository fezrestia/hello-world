namespace hoge {
    export class foo{

        // Get jQuery object.
        constructor(private element:JQuery){
        }

        public color(color:string) {
            this.element.css('color', color);
        }
    }
}


$(function() {
    var foo = new hoge.foo($('div'));
    foo.color('blue')
});

