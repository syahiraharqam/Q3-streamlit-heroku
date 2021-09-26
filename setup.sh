mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = truen\n\
enableCORS=falsen\n\
port = $PORTn\n\
" > ~/.streamlit/config.toml