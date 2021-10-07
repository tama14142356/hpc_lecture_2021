#!/bin/bash
pyright_setting_file="pyrightconfig.json"

echo "{" > "$pyright_setting_file"
pyenv_root=$(pyenv root)
version_name=$(pyenv version-name)
echo -e "\t\"venvPath\": \""$pyenv_root"/versions/\"," >> "$pyright_setting_file"
echo -e "\t\"venv\": \""$version_name"\"" >> "$pyright_setting_file"
echo "}" >> "$pyright_setting_file"
