#include <stdio.h>
#include <sys/socket.h>
#include <sys/un.h>

int main(void) {
    printf("TraceLog: server: main() : E\n");

    int result = 0;

    // Create server socket file discriptor.
    int socketfd = socket(
            AF_UNIX, // Local connection.
            SOCK_STREAM, // Bi-directional access, bite stream.
            0); // Default protocol.
    if (socketfd == -1) {
        printf("TraceLog: server: socketfd != 0\n");
        return -1;
    }

    // Struct for socket address.
    struct sockaddr_un addr = { 0 }; // 0 init.
    addr.sun_family = AF_UNIX; // Fixed.
    strcpy(addr.sun_path, "socket_file"); // Common socket file.

    // Remove old socket.
    remove(addr.sun_path);

    // Bind.
    result = bind(
            socketfd, // Bind target socket.
            (struct sockaddr*) &addr, // Socket address.
            sizeof(struct sockaddr_un)); // Address size.
    if (result != 0) {
        printf("TraceLog: server: bind failed.\n");
        close(socketfd);
        return -1;
    }

    // Mark passive socket.
    result = listen(
            socketfd, // Target socket.
            1); // Max connection count.
    if (result != 0) {
        printf("TraceLog: server: listen failed.\n");
        close(socketfd);
        return -1;
    }

    // Wait for connection and receive.
    while(1) {
        // Wait for client connection.
        int fd = accept(
                socketfd, // Target socket.
                NULL, // Client address.
                NULL); // Client address size.
        if (fd == -1) {
            printf("TraceLog: server: accept failed.\n");
            close(socketfd);
            return -1;
        }

        // Receive.
        char buffer[256];
        int size = read(
                fd, // Accepted fd.
                buffer, // Receive buffer.
                sizeof(buffer) - 1); // Receive size. Last byte is null char.
        if (size == -1) {
            printf("TraceLog: server: read failed.\n");
            close(socketfd);
            return -1;
        }

        // Output received message.
        buffer[size] = '\0';
        printf("TraceLog: server: Received Msg = %s\n", buffer);

        // Close client fd.
        result = close(fd); // Closed client connection fd.
        if (result != 0) {
            printf("TraceLog: server: close fd failed.\n");
            close(socketfd);
            return -1;
        }

    } // while(true)

    // Close server socket.
    close(socketfd);

    printf("## server main() : X\n");
    return 0;
}

