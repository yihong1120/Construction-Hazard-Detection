🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Streaming Web Frontend Example

This project demonstrates a web frontend for streaming camera feeds, built using HTML, CSS, JavaScript, and Vite. The frontend interacts with a backend API to fetch and display camera streams and related metadata.

## Usage

To get started with the project, follow these steps:

### 1. Install Node.js and npm

Ensure you have Node.js and npm installed. You can download and install them from the [official Node.js website](https://nodejs.org/). The recommended way is to use the LTS (Long Term Support) version.

#### Installation on Windows

1. **Download the Windows installer**:
    - Go to the [Node.js website](https://nodejs.org/).
    - Click on the LTS version to download the installer.

2. **Run the installer**:
    - Open the downloaded `.msi` file.
    - Follow the prompts in the installer. Accept the license agreement, choose the installation path, and ensure that the option to add Node.js to PATH is checked.

3. **Verify the installation**:
    - Open Command Prompt or PowerShell.
    - Run the following commands to check if Node.js and npm are installed correctly:
      ```sh
      node -v
      npm -v
      ```
    - These commands should print the versions of Node.js and npm installed on your system.

4. **Update npm (optional but recommended)**:
    - Sometimes, the version of npm included with Node.js might be outdated. You can update npm to the latest version by running:
      ```sh
      npm install -g npm
      ```

#### Installation on macOS

1. **Download the macOS installer**:
    - Go to the [Node.js website](https://nodejs.org/).
    - Click on the LTS version to download the installer.

2. **Run the installer**:
    - Open the downloaded `.pkg` file.
    - Follow the prompts in the installer.

3. **Alternatively, use Homebrew**:
    - If you prefer using a package manager, you can install Node.js via Homebrew:
      ```sh
      brew install node
      ```

4. **Verify the installation**:
    - Open Terminal.
    - Run the following commands to check if Node.js and npm are installed correctly:
      ```sh
      node -v
      npm -v
      ```

#### Installation on Linux

For Debian-based distributions (e.g., Ubuntu), you can use the following commands:

1. **Add NodeSource repository**:
    ```sh
    curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
    ```

2. **Install Node.js and npm**:
    ```sh
    sudo apt-get install -y nodejs
    ```

3. **Verify the installation**:
    - Open Terminal.
    - Run the following commands to check if Node.js and npm are installed correctly:
      ```sh
      node -v
      npm -v
      ```

For other distributions, please refer to the [Node.js installation guide](https://nodejs.org/en/download/package-manager/).

### 2. Clone the repository

Clone the repository to your local machine using Git:
```sh
git clone https://your-gitea-server.com/username/repository.git
cd repository/examples/streaming_web/frontend
```

### 3. Install dependencies

Navigate to the project directory and install the required dependencies by running:
```sh
npm install
```

### 4. Start the development server

Start the Vite development server on port 8888 by running:
```sh
npm run dev
```
You should see output indicating that the server is running. Open your browser and navigate to `http://localhost:8888` to view the application.

### 5. Build for production

To build the project for production, run:
```sh
npm run build
```
The built files will be output to the `dist` directory. You can then serve these files using a static file server or deploy them to a web server.

## Features

- **Responsive Design**: The frontend is designed to be responsive and adapts to various screen sizes, ensuring a good user experience on both desktop and mobile devices.
- **Dynamic Content**: Camera labels and streams are fetched dynamically from the backend API and displayed on the frontend.
- **WebSocket Integration**: Real-time updates are handled via WebSocket connections, ensuring that the camera streams and warnings are updated live.
- **Error Handling**: The frontend includes error handling to manage issues such as missing URL parameters and WebSocket connection errors.
- **Accessibility**: The design includes considerations for accessibility, such as alternative text for images and high contrast mode adjustments.

## Configuration

The project configuration is managed via the `vite.config.js` file. Key configuration options include:

- **Root Directory**: The root directory for the project is set to `public`.
- **Development Server**: The development server is configured to run on port 8888, with a proxy setup to forward API requests to the backend server running on `http://127.0.0.1:8000`.
- **Build Output**: The build output directory is set to `dist`, located outside of the `public` directory. The Rollup options specify the entry points for the main, label, and camera HTML files.

### Example `vite.config.js`

```javascript
import { defineConfig } from 'vite';

// Export the Vite configuration object
export default defineConfig({
  // Set the root directory for the project to 'public'
  root: 'public',

  server: {
    // Define the port for the development server
    port: 8888,

    proxy: {
      // Configure proxy settings for the '/api' prefix
      '/api': {
        // The backend server to forward API requests to
        target: 'http://127.0.0.1:8000',

        // Enable changing the origin of the host header to the target URL
        changeOrigin: true,

        // Enable WebSocket proxying
        ws: true,
      },
    },
  },

  build: {
    outDir: '../dist', // Output directory outside of public
    rollupOptions: {
      input: {
        main: './public/index.html', // Main entry
        label: './public/label.html', // Additional entry
        camera: './public/camera.html', // Additional entry
      },
    },
  },
});
```

## Package Information

The project uses the following `package.json` configuration:

```json
{
    "name": "streaming-web-frontend",
    "version": "1.0.0",
    "description": "Frontend for the streaming web application with FastAPI backend.",
    "main": "index.js",
    "scripts": {
        "dev": "vite",
        "build": "vite build",
        "preview": "vite preview"
    },
    "keywords": [],
    "author": "yihong1120",
    "license": "AGPL-3.0-only",
    "dependencies": {
        "axios": "^1.5.0"
    },
    "devDependencies": {
        "vite": "^5.4.11"
    }
}
```

## File Structure

The project is organised as follows:

- **public/**: Contains the HTML, CSS, and JavaScript files for the frontend.
  - **index.html**: Main page displaying camera labels.
  - **label.html**: Page displaying details for a specific label.
  - **camera.html**: Page displaying the live stream for a specific camera.
  - **css/**: Directory containing the stylesheet (`styles.css`).
  - **js/**: Directory containing the JavaScript files (`index.js`, `label.js`, `camera.js`).
- **package.json**: Configuration file for packages.
- **vite.config.js**: Configuration file for Vite.

### Scripts

- **`dev`**: Starts the Vite development server.
- **`build`**: Builds the project for production.
- **`preview`**: Previews the built project.

### Dependencies

- **`axios`**: A promise-based HTTP client for the browser and Node.js.

### DevDependencies

- **`vite`**: A build tool that aims to provide a faster and leaner development experience for modern web projects.

## Additional Information

For more details on how to use and extend this project, please refer to the comments within the source code files. The comments provide explanations for the various sections and functionalities implemented in the project.

### Development Tips

- **Hot Module Replacement (HMR)**: Vite supports HMR, which means that changes you make to your code will be reflected in the browser without needing a full page reload. This speeds up development and provides immediate feedback.
- **Debugging**: Use browser developer tools to debug JavaScript code. You can set breakpoints, inspect variables, and step through code to understand how it works.
- **Linting and Formatting**: Consider using tools like ESLint and Prettier to maintain code quality and consistency. These tools can be integrated into your development workflow to automatically check and format your code.

### Deployment

To deploy the built files, you can use any static file server or web hosting service. Here are a few options:

- **Netlify**: A popular choice for deploying static websites. You can connect your Git repository, and Netlify will automatically build and deploy your site.
- **Vercel**: Another popular platform for deploying static sites and serverless functions. It also integrates well with Git repositories.
- **GitHub Pages**: If your repository is hosted on GitHub, you can use GitHub Pages to serve your static site directly from the repository.