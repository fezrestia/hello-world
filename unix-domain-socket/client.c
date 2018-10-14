#include <stdio.h>
#include <sys/socket.h>
#include <sys/un.h>

#define MESSAGE "HELLO WORLD !"

void on_error(char* msg, int socket_fd) {
    if (socket_fd != -1) {
        close(socket_fd);
    }
    printf("TraceLog: client: %s\n", msg);
}

int main(void) {
    printf("TraceLog: client: main() : E\n");

    int err = 0;
    int socket_fd = -1;

    // Create client socket.
    socket_fd = socket(
            AF_UNIX, // Local connection.
            SOCK_STREAM, // Bi-directional access, bite stream.
            0); // Default protocol.
    if (socket_fd == -1) {
        on_error("socket_fd != 0", socket_fd);
        return -1;
    }

    // Struct for socket address.
    struct sockaddr_un addr = { 0 }; // 0 init.
    addr.sun_family = AF_UNIX; // Fixed.
    strcpy(addr.sun_path, "socket_file"); // Common socket file.

    // Connect.
    err = connect(
            socket_fd, // Bind target socket.
            (struct sockaddr*) &addr, // Socket address.
            sizeof(struct sockaddr_un)); // Address size.
    if (err != 0) {
        on_error("connect failed.", socket_fd);
        return -1;
    }

    // Send message.
    err = write(
            socket_fd, // Target socket.
            MESSAGE, // Message.
            strlen(MESSAGE)); // Message size.
    if (err == -1) {
        on_error("write failed.", socket_fd);
        return -1;
    }

    // Close client socket.
    close(socket_fd);

    printf("TraceLog: client: main() : X\n");
    return 0;
}

