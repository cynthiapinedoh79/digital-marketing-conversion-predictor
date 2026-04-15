import streamlit as st


class MultiPage:
    """
    Generates multiple Streamlit pages using an object-oriented approach.
    Based on the Code Institute Churnometer walkthrough project pattern.
    """

    def __init__(self, app_name: str) -> None:
        self.pages = []
        self.app_name = app_name

        st.set_page_config(
            page_title=self.app_name,
            page_icon=":bar_chart:",
            layout="wide"
        )

    def add_page(self, title: str, func) -> None:
        self.pages.append({"title": title, "function": func})

    def run(self):
        st.sidebar.title(self.app_name)
        st.sidebar.markdown("---")

        page = st.sidebar.radio(
            "Navigation",
            self.pages,
            format_func=lambda page: page["title"]
        )

        st.sidebar.markdown("---")
        st.sidebar.info(
            "This dashboard was built as part of the "
            "Code Institute Diploma in Full Stack Software Development "
            "(Predictive Analytics) — Portfolio Project 5."
        )

        page["function"]()
