FROM node:ubuntu
COPY . /app
WORKDIR /app
CMD node app.js
