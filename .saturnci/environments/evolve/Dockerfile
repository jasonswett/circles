FROM rust:1.91-slim

RUN apt-get update && apt-get install -y \
  ruby \
  git \
  && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch v1.0.2 https://github.com/saturnci/saturnci-sdk /tmp/saturnci-sdk \
  && cd /tmp/saturnci-sdk \
  && gem build saturnci-sdk.gemspec \
  && gem install saturnci-sdk-*.gem \
  && rm -rf /tmp/saturnci-sdk

WORKDIR /app

ENTRYPOINT ["ruby", "-rsaturnci-sdk"]
