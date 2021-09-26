mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"syahiraharqam@gmail.com"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = truen\n\
enableCORS=falsen\n\
port = $PORTn\n\
" > ~/.streamlit/config.toml