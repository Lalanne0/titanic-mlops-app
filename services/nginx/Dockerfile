FROM nginx:alpine

ARG NGINX_HOST=localhost

# Generate a self-signed TLS certificate
RUN apk add --no-cache openssl && \
    mkdir -p /etc/nginx/certs && \
    openssl req -x509 -nodes -days 365 \
        -newkey rsa:2048 \
        -keyout /etc/nginx/certs/server.key \
        -out /etc/nginx/certs/server.crt \
        -subj "/C=FR/ST=IDF/L=Paris/O=TitanicMLOps/CN=${NGINX_HOST}" \
        2>/dev/null

COPY services/nginx/nginx.conf /etc/nginx/nginx.conf

EXPOSE 80 443
