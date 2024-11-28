# Use the official Nginx image as the base
FROM nginx:alpine

# Copy the built frontend files to Nginx's default static file directory
COPY dist/ /usr/share/nginx/html/

# Copy the Nginx configuration file
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
