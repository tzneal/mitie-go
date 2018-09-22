FROM ubuntu:bionic


RUN apt-get update && apt-get -y install git gcc g++ gfortran cmake libopenblas-dev liblapack-dev ruby ruby-dev rubygems build-essential wget

ENV MITIE_VER 0.6
RUN git clone https://github.com/mit-nlp/MITIE.git
RUN cd MITIE && git checkout v${MITIE_VER}

WORKDIR MITIE/mitielib/build
RUN cmake ..
RUN cmake --build . --config Release --target install

WORKDIR /
RUN mkdir -p /dist/usr/lib
RUN cp -a MITIE/mitielib/include /dist/usr
RUN cp MITIE/mitielib/*.so /dist/usr/lib

RUN gem install --no-ri --no-rdoc fpm
RUN wget https://github.com/mit-nlp/MITIE/releases/download/v0.4/MITIE-models-v0.2.tar.bz2
RUN mkdir -p /dist/usr/lib /dist/usr/share/mitie
RUN tar -xvf MITIE-models-v0.2.tar.bz2 && mv MITIE-models/english/ /dist/usr/share/mitie
RUN fpm -s dir -t deb -n libmitie --depends libopenblas-dev --depends liblapack-dev -v ${MITIE_VER} -C /dist ./
