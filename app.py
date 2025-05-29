import streamlit as st
from transformers import AutoTokenizer


def load_models_from_file(path: str = "models.txt") -> list[str]:
    """
    Reads model IDs from models.txt, one per line.
    """
    try:
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        st.sidebar.error(
            f"Model list file '{path}' not found.\nRun fetch_models.py first to generate it."
        )
        return []


@st.cache_data
def get_tokenizer(model_name: str):
    """
    Initializes and caches the tokenizer for the selected model.
    Falls back to GPT-2 if unavailable.
    """
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except Exception:
        print("=====================모델 cannot load ====================")
        return AutoTokenizer.from_pretrained("gpt2")


def main():
    st.title("LLM Token & Character Counter")
    st.sidebar.header("Settings")

    # Load full model list
    model_list = load_models_from_file()
    if not model_list:
        return

    # Add empty option for initial prompt
    options = [""] + model_list
    selected_model = st.sidebar.selectbox(
        "Choose an LLM model", options, index=0
    )

    # Prompt user to select a model
    if not selected_model:
        st.sidebar.info("Start typing to search and select a model.")
        return

    # Load tokenizer for the selected model
    tokenizer = get_tokenizer(selected_model)

    # Text input area
    user_input = st.text_area("Enter your text here:", height=200)

    # Button to trigger counting
    if st.button("Count"):  # or "Counting" as label
        if user_input:
            char_count = len(user_input)
            encoded = tokenizer(user_input, return_attention_mask=False)
            token_count = len(encoded["input_ids"])
            st.write(f"**Character count:** {char_count}")
            st.write(f"**Token count:** {token_count}")
        else:
            st.warning("Please enter some text to count.")


if __name__ == "__main__":
    main()
