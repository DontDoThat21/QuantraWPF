using System;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Quantra
{
    /// <summary>
    /// Simple WebSocket wrapper class for handling WebSocket connections
    /// </summary>
    public class WebSocket : IDisposable
    {
        private ClientWebSocket _webSocket;
        private CancellationTokenSource _cancellationTokenSource;
        private bool _isConnected = false;
        private Task _receiveTask;

        /// <summary>
        /// Event raised when a message is received
        /// </summary>
        public event EventHandler<string> MessageReceived;

        /// <summary>
        /// Event raised when the connection is closed
        /// </summary>
        public event EventHandler ConnectionClosed;

        /// <summary>
        /// Gets whether the socket is currently connected
        /// </summary>
        public bool IsConnected => _isConnected && _webSocket?.State == WebSocketState.Open;

        /// <summary>
        /// Creates a new WebSocket instance
        /// </summary>
        public WebSocket()
        {
            _webSocket = new ClientWebSocket();
            _cancellationTokenSource = new CancellationTokenSource();
        }

        /// <summary>
        /// Connects to a WebSocket server
        /// </summary>
        /// <param name="url">URL of the WebSocket server</param>
        /// <returns>True if connected successfully, otherwise false</returns>
        public async Task<bool> ConnectAsync(string url)
        {
            try
            {
                if (_webSocket.State != WebSocketState.None && _webSocket.State != WebSocketState.Closed)
                {
                    // Already connecting or connected
                    return _webSocket.State == WebSocketState.Open;
                }

                // Create a new WebSocket if needed
                if (_webSocket.State == WebSocketState.Closed)
                {
                    _webSocket = new ClientWebSocket();
                    _cancellationTokenSource = new CancellationTokenSource();
                }

                // Connect to the WebSocket server
                await _webSocket.ConnectAsync(new Uri(url), _cancellationTokenSource.Token);
                _isConnected = true;

                // Start receiving messages
                _receiveTask = StartReceiving();

                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"WebSocket connection error: {ex.Message}");
                _isConnected = false;
                return false;
            }
        }

        /// <summary>
        /// Sends a message through the WebSocket
        /// </summary>
        /// <param name="message">Message to send</param>
        /// <returns>True if sent successfully, otherwise false</returns>
        public async Task<bool> SendAsync(string message)
        {
            try
            {
                if (!IsConnected)
                {
                    return false;
                }

                byte[] buffer = Encoding.UTF8.GetBytes(message);
                await _webSocket.SendAsync(
                    new ArraySegment<byte>(buffer),
                    WebSocketMessageType.Text,
                    true,
                    _cancellationTokenSource.Token);

                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"WebSocket send error: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Closes the WebSocket connection
        /// </summary>
        public async Task CloseAsync()
        {
            try
            {
                if (_webSocket.State == WebSocketState.Open)
                {
                    // Cancel any ongoing operations
                    _cancellationTokenSource.Cancel();

                    // Close the connection gracefully
                    await _webSocket.CloseAsync(
                        WebSocketCloseStatus.NormalClosure,
                        "Connection closed by client",
                        CancellationToken.None);
                }

                _isConnected = false;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"WebSocket close error: {ex.Message}");
            }
        }

        /// <summary>
        /// Starts receiving messages from the WebSocket
        /// </summary>
        private async Task StartReceiving()
        {
            try
            {
                byte[] buffer = new byte[4096];
                while (_webSocket.State == WebSocketState.Open)
                {
                    // Create a string builder to accumulate the message
                    StringBuilder messageBuilder = new StringBuilder();
                    WebSocketReceiveResult result;

                    do
                    {
                        // Receive a chunk of the message
                        result = await _webSocket.ReceiveAsync(
                            new ArraySegment<byte>(buffer),
                            _cancellationTokenSource.Token);

                        // Append the chunk to the message
                        if (result.MessageType == WebSocketMessageType.Text)
                        {
                            string chunk = Encoding.UTF8.GetString(buffer, 0, result.Count);
                            messageBuilder.Append(chunk);
                        }
                    }
                    while (!result.EndOfMessage);

                    // Process the complete message
                    if (result.MessageType == WebSocketMessageType.Text)
                    {
                        string message = messageBuilder.ToString();
                        MessageReceived?.Invoke(this, message);
                    }
                    else if (result.MessageType == WebSocketMessageType.Close)
                    {
                        await CloseAsync();
                        ConnectionClosed?.Invoke(this, EventArgs.Empty);
                        break;
                    }
                }
            }
            catch (OperationCanceledException)
            {
                // Expected when cancellation is requested
            }
            catch (WebSocketException ex)
            {
                Console.WriteLine($"WebSocket receive error: {ex.Message}");
                _isConnected = false;
                ConnectionClosed?.Invoke(this, EventArgs.Empty);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"WebSocket error: {ex.Message}");
                _isConnected = false;
                ConnectionClosed?.Invoke(this, EventArgs.Empty);
            }
        }

        /// <summary>
        /// Disposes of resources
        /// </summary>
        public void Dispose()
        {
            // Cancel any ongoing operations
            _cancellationTokenSource?.Cancel();

            // Close and dispose the WebSocket
            _webSocket?.Dispose();
            _webSocket = null;

            // Dispose the cancellation token source
            _cancellationTokenSource?.Dispose();
            _cancellationTokenSource = null;

            _isConnected = false;
        }
    }
}