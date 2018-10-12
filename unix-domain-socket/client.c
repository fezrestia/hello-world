#include <stdio.h>
#include <sys/socket.h>
#include <sys/un.h>

#define MESSAGE "HELLO WORLD !"

int main(void) {
    printf("TraceLog: client: main() : E\n");

    int result = 0;

    // Create client socket.
    int socketfd = socket(
            AF_UNIX, // Local connection.
            SOCK_STREAM, // Bi-directional access, bite stream.
            0); // Default protocol.
    if (socketfd == -1) {
        printf("TraceLog: client: socketfd != 0\n");
        return -1;
    }

    // Struct for socket address.
    struct sockaddr_un addr = { 0 }; // 0 init.
    addr.sun_family = AF_UNIX; // Fixed.
    strcpy(addr.sun_path, "socket_file"); // Common socket file.

    // Connect.
    result = connect(
            socketfd, // Bind target socket.
            (struct sockaddr*) &addr, // Socket address.
            sizeof(struct sockaddr_un)); // Address size.
    if (result != 0) {
        printf("TraceLog: client: connect failed.\n");
        close(socketfd);
        return -1;
    }

    // Send message.
    result = write(
            socketfd, // Target socket.
            MESSAGE, // Message.
            strlen(MESSAGE)); // Message size.
    if (result == -1) {
        printf("TraceLog: client: write failed.\n");
        close(socketfd);
        return -1;
    }

    // Close client socket.
    close(socketfd);

    printf("TraceLog: client: main() : X\n");
    return 0;
}

