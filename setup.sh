mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = 8080\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
