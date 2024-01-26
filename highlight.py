import gradio as gr

from gradio_highlightedtextbox import HighlightedTextbox


def convert_tagged_text_to_highlighted_text(
    tagged_text: str, tag_id: str, tag_open: str, tag_close: str
) -> list[tuple[str, str | None]]:
    return HighlightedTextbox.tagged_text_to_tuples(
        tagged_text, tag_id, tag_open, tag_close
    )


with gr.Blocks() as demo:
    tag_id = gr.Textbox(
        "Potential issue",
        label="Tag ID",
        show_label=True,
        info="Insert a tag ID to use in the highlighted textbox.",
    )
    tag_open = gr.Textbox(
        "<h>",
        label="Tag open",
        show_label=True,
        info="Insert a tag to mark the beginning of a highlighted section.",
    )
    tag_close = gr.Textbox(
        "</h>",
        label="Tag close",
        show_label=True,
        info="Insert a tag to mark the end of a highlighted section.",
    )
    with gr.Row():
        tagged = gr.Textbox(
            "It is not something to be ashamed of: it is no different from the <h>personal fears</h> and <h>dislikes</h> of other things that <h>very many people</h> have.",
            interactive=True,
            label="Input",
            show_label=True,
            info="Insert a text with <h>...</h> tags to mark spans that will be highlighted.",
        )
        high = HighlightedTextbox(
            interactive=True,
            label="Output",
            info="Highlighted textbox.",
            show_legend=True,
            show_label=True,
            legend_label="Legend:",
            show_legend_label=True,
        )
    button = gr.Button("Submit")
    button.click(
        fn=convert_tagged_text_to_highlighted_text,
        inputs=[tagged, tag_id, tag_open, tag_close],
        outputs=high,
    )
    # Initialization does not work
    high = HighlightedTextbox(
        convert_tagged_text_to_highlighted_text(
            tagged.value, tag_id.value, tag_open.value, tag_close.value
        ),
        interactive=True,
        label="Does not work",
        show_label=True,
    )


demo.launch()