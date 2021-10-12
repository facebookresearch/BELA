mkdir /fsx/kassner/OSCAR/
mkdir /fsx/kassner/OSCAR/metadata/

./dbxcli-linux-amd64 ls -l "OSCAR/metadata"|awk -F"ago" '{print $2}'|sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'|xargs -I{} ./dbxcli-linux-amd64 get {} /fsx/kassner/{}
