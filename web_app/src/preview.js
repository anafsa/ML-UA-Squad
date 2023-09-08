import { html, css, svg, LitElement } from 'https://cdn.jsdelivr.net/gh/lit/dist@2/core/lit-core.min.js';

export class PreviewThumb extends LitElement {
	static styles = css`
		img.thumbnail {
			width: 12vw;
			height: 12vw;
		}
		
		div.title {
			border: 0.05em solid black;
			width: 12vw;
			display: flex;
			justify-content: center;
			background: #aaa9;
		}
  	`;

	static properties = {
		title: { type: String },
		src: { type: String }
	};

	constructor() {
		super();
	}

	render() {
		return html`
			<div class="title">
				<div> ${this.title.slice(0, 12) + "..."} </div>
			</div>
			<img src=${this.src} class="thumbnail" />
		`
	}
}
customElements.define('preview-thumb', PreviewThumb);
