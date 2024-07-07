🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Line Chatbot Example

This repository provides an example of creating a LINE chatbot using Flask and the LINE Messaging API. The example demonstrates how to set up a basic chatbot that responds with the same text message it receives.

## Usage

### Setting Up Environment Variables

Before starting your chatbot, you need to set up the necessary environment variables:

1. `LINE_CHANNEL_ACCESS_TOKEN`: Your LINE channel access token.
2. `LINE_CHANNEL_SECRET`: Your LINE channel secret.

These can be obtained from the LINE Developer Console after creating your LINE bot. Follow the steps below to retrieve your credentials:

#### LINE Developer Console Setup

1. Go to the [LINE Developer Console](https://developers.line.biz/console/).
2. Sign in with your LINE account.
3. Create a new provider and bot if you haven't already.
4. Select your bot channel from the list.
5. In the channel settings, find the Channel Secret and Channel Access Token section.
6. Copy the Channel Secret and generate a Channel Access Token if you haven't already.

You can set these variables in your environment or replace `'YOUR_LINE_CHANNEL_ACCESS_TOKEN'` and `'YOUR_LINE_CHANNEL_SECRET'` in the `line_bot.py` script with your actual LINE channel access token and secret.

### Running the Chatbot

To run the chatbot, first ensure you have Flask installed, or install it using pip:

```bash
pip install Flask
```

Then, run the `line_bot.py` script:

```bash
python line_bot.py
```

The Flask application will start, and your chatbot will be running locally.

### Webhook Setup

To allow the LINE platform to communicate with your chatbot, you need to set up a webhook URL. This URL should point to the `/callback` endpoint where your Flask application is running.

1. Go back to the LINE Developer Console and select your bot.
2. In the settings, find the Webhook settings section.
3. Set the Webhook URL to the URL where your Flask app is running followed by `/callback`. For example, `https://yourappname.ngrok.io/callback`.
4. Verify the webhook.

If you're testing locally, you can use services like [ngrok](https://ngrok.com/) to expose your local server to the internet.

After setting up your webhook, any messages sent to your LINE bot will be echoed back to the sender.

## Features

- **Flask Web Application**: Uses Flask to create a web server that can handle requests from LINE.
- **LINE Messaging API**: Utilizes the LINE Messaging API to receive and respond to messages.
- **Webhook Handling**: Includes an example of how to handle webhook events from LINE.
- **Message Echo**: The chatbot echoes back any text messages it receives.

## Configuration

You can customize the chatbot's responses or add more complex functionalities by modifying the `handle_message` function in `line_bot.py`.

Remember to secure your webhook endpoint, especially if you deploy your chatbot to production.

## Deployment

For deployment, you can use any cloud platform that supports Python and Flask, such as Heroku, AWS, or Google Cloud. Remember to update your webhook URL in the LINE Developer Console with your deployed application's URL.

Ensure your environment variables are set up correctly in your deployment environment.

## Further Development

This example provides a basic setup. For more advanced features like rich messages, quick replies, and template messages, refer to the [LINE Messaging API documentation](https://developers.line.biz/en/docs/messaging-api/).

Explore adding functionalities such as user authentication, database integration, and more complex conversation flows to enhance your chatbot.
