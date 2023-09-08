
let stageID = 0;
let sceneID = 0;
let action = "";
let data = {};

d3.select("#scene-back").on("click", function (e, d) {
	stageID--;
	action = "back";
	loadUI();
});
d3.select("#scene-forward").on("click", function (e, d) {
	stageID++;
	action = "forward";
	loadUI();
});

function loadUI() {
	let params = {};
	let hasChangeScene = false;

	// change scene
	if (stageID >= data.story[sceneID].text.length) {
		sceneID++;
		if (sceneID >= data.story.length) {
			sceneID = data.story.length - 1;
		}
		stageID = 0;
		hasChangeScene = true;
	}
	if (stageID < 0) {
		sceneID--;
		if (sceneID < 0) { // avoid async bugs
			sceneID = 0;
		}
		stageID = data.story[sceneID].text.length - 1;
		hasChangeScene = true;
	}
	params.text = data.story[sceneID].text[stageID];
	params.image = data.story[sceneID].image;
	params.action = action;
	params.changeScene = hasChangeScene;

	params.stage = "middle";

	params.stage = sceneID == 0 && stageID == 0 ? "first" : params.stage;
	params.stage = sceneID == data.story.length - 1 &&
		stageID == data.story[sceneID].text.length - 1 ? "last" : params.stage;
	updateUI(params);
}

function updateUI(x) {

	if (x.stage == "last") {
		d3.select("#scene-forward").classed("arrow-disabled", true);
	} else {
		d3.select("#scene-forward").classed("arrow-disabled", false);
	}
	if (x.stage == "first") {
		d3.select("#scene-back").classed("arrow-disabled", true);
	} else {
		d3.select("#scene-back").classed("arrow-disabled", false);
	}
	if (x.changeScene) {

	}
	d3.select("#main-text").text(x.text);
	d3.select("#main-image").attr("src", `${x.image}`);
}

function setupMenu() {
	const drawer = document.querySelector('.drawer-overview');
	const openButton = d3.select("#menu").node();
	const closeButton = drawer.querySelector('sl-button[variant="primary"]');

	openButton.addEventListener('click', () => drawer.show());
	// closeButton.addEventListener('click', () => drawer.hide());
}

function setupSound() {
	d3.selectAll("#sound-off").style("display", "none");
	d3.selectAll(".sound").on("click", function () {
		d3.selectAll(".sound").style("display", "block")
		this.style.display = "none";
	});
}

function loadStory() {
	const urlParams = new URLSearchParams(window.location.search);
	const myParam = urlParams.get('q');
	
	d3.json("finalStories/" + myParam).then(ss => {
		data = ss;
		for (let i = 0; i < data.story.length; i++) {
			data.story[i].text = data.story[i].text.split(".");
			data.story[i].text.pop()
		}
		console.log(data);
		loadUI();
		setupMenu();
		setupSound();
	})
}

loadStory();