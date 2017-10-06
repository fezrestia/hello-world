// Start up notifyer.
function on_d3_sample_loaded() {
    alert("D3 Sample JavaScript is Loaded.");
}


function setup(json) {
    var margin = { top: 20, right: 20, bottom: 30, left: 50 },
            width = 640 - margin.left - margin.right,
            height = 480 - margin.top - margin.bottom;

    var scale = d3.scale.linear()
            .domain([0, 1000])
            .range([0, 1000])

    var x = d3.scale.ordinal().rangeRoundBands([0, width], .1);
    var y = d3.scale.linear().range([height, 0]);

    var xAxis = d3.svg.axis()
            .scale(x)
            .orient("bottom");
    var yAxis = d3.svg.axis()
            .scale(y)
            .orient("left");

    var svg = d3.select("#d3_bar_sample").append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    d3.json(json, function(error, data) {
            data.forEach(function(d) {
                    d.score = +d.score;
            });

            x.domain(data.map(function(d) { return d.label; }));
            y.domain([0, d3.max(data, function(d) { return d.score; })]);

            svg.append("g")
                    .attr("class", "x axis")
                    .attr("transform", "translate(" + (20 - margin.left - margin.right) + "," + height + ")")
                    .call(xAxis);

            svg.append("g")
                    .attr("class", "y axis")
                    .call(yAxis)
                    .append("text")
                    .attr("transform", "rotate(-90)")
                    .attr("y", 6)
                    .attr("dy", ".71em")
                    .style("text-anchor", "end")
                    .text("score");

            svg.selectAll(".bar")
                    .data(data)
                    .enter().append("rect")
                    .attr("class", "bar")
                    .attr("x", function(d) { return x(d.label); })
                    .attr("width", x.rangeBand())
                    .attr("y", function(d) { return y(d.score); })
                    .attr("height", function(d) { return height - y(d.score); })
    });
}


function setup_dependency_wheel(json) {
    var margin = { top: 20, right: 20, bottom: 20, left: 20 };
    var width = 640 - margin.left - margin.right;
    var height = 640 - margin.top - margin.bottom;

    var svg = d3.select("#d3_dependency_wheel").append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom);

    var chart = d3.chart.dependencyWheel();

    d3.json(json, function(error, data) {
        svg
                .datum(data)
                .call(chart);
    });








}

