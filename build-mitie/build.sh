docker build -t mitie-build .
ID=$(docker create mitie-build)

rm libmitie_0.6_amd64.deb
docker cp $ID:/libmitie_0.6_amd64.deb .
docker rm $ID
