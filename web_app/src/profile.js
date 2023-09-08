let data = {};

const urlParams = new URLSearchParams(window.location.search);
const json_path = urlParams.get('q');

function loadUI() {
	d3.select("#title").text(data.title);
	d3.selectAll(".tag").data(data.tags).join(".tag").text(d => d);
	d3.select("#main-image").attr("src", data.thumbnail);
	d3.select("#start").on("click", function(e,d) {
		location.href = 'http://127.0.0.1:8000/scene.html?q='+json_path;
	});
	d3.select("#summary").text(data.summary)
}

function loadStory() {
	
	d3.json("finalStories/" + json_path).then(ss => {
		data = ss;
		loadUI();
	})
}

loadStory();