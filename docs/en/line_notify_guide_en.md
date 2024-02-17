🇬🇧 [English](../en/line_notify_guide_en.md) | 🇹🇼 [繁體中文](../zh/line_notify_guide_zh.md)

# How to Use LINE Notify

This guide provides a straightforward explanation on how to use LINE Notify to send messages from various applications to your LINE account. LINE Notify allows you to receive notifications from web services and applications directly on LINE upon configuration. Before proceeding, ensure you have generated your personal LINE Notify token.

## Prerequisites

- A LINE account
- Personal LINE Notify token (referred to as `YOUR_LINE_NOTIFY_TOKEN` in this guide)

## Step 1: Generate a LINE Notify Token

1. Visit the [LINE Notify website](https://notify-bot.line.me/en/).
2. Sign in with your LINE credentials.
3. Navigate to the "My Page" section.
4. Click on the "Generate token" button.
5. Choose the desired service and provide a name for the token.
6. Click on "Generate". Remember to save your token securely.

## Step 2: Install Required Tools

For this tutorial, we will be using `curl` to demonstrate how to send notifications. Ensure you have `curl` installed on your system. Most Unix-like operating systems, including Linux and macOS, come with `curl` pre-installed.

## Step 3: Sending a Message with LINE Notify

To send a notification to your LINE, use the following `curl` command in your terminal. Replace `YOUR_LINE_NOTIFY_TOKEN` with the token you generated in Step 1.

```bash
curl -X POST https://notify-api.line.me/api/notify \
     -H 'Authorization: Bearer YOUR_LINE_NOTIFY_TOKEN' \
     -F 'message=Hello! This is a test message from LINE Notify.'
```

## Step 4: Using Python to Automate Notifications

To automate sending notifications with LINE Notify, you can use the provided Python example code in `src/line_notifier.py`. This script demonstrates how to programmatically send messages using the LINE Notify API.

### Prerequisites for Running the Python Script:

- Python installed on your system.
- The `requests` library installed. You can install it using pip: `pip install requests`.

### How to Use the Python Script:

1. Ensure you replace `'YOUR_LINE_NOTIFY_TOKEN'` in the script with your actual LINE Notify token.
2. Customize the message you wish to send. The example script sends a timestamped message warning about a specific event.
3. Execute the script by running `python src/line_notifier.py` in your terminal.

### Script Explanation:

- The script defines a function `send_line_notification` that takes a LINE Notify token and a message as arguments.
- It sends a POST request to the LINE Notify API with the specified message.
- The function returns the HTTP status code from the API, indicating the success or failure of the notification delivery.

```python
# Example usage of the send_line_notification function:
line_token = 'YOUR_LINE_NOTIFY_TOKEN'  # Replace with your actual LINE Notify token
message = 'Hello, this is a test message.'
status = send_line_notification(line_token, message)
print(status)  # Prints the HTTP status code (e.g., 200 for success)
```

## Step 5: Verifying Receipt

Upon successful execution, check your LINE application. You should receive the message sent from the script or the `curl` command.

## Troubleshooting

- **Invalid Token**: Ensure your token is correctly inputted. If it continues to fail, regenerate a token and try again.
- **No Message Received**: Check your internet connection and ensure LINE is not restricted on your network.

For more detailed information and advanced usage, please refer to the [official LINE Notify documentation](https://notify-bot.line.me/doc/en/).