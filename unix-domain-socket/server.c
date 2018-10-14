#include <stdio.h>
#include <sys/socket.h>
#include <sys/un.h>

void on_error(char* msg, int socket_fd) {
    if (socket_fd != -1) {
        close(socket_fd);
    }
    printf("TraceLog: server: %s\n", msg);
}

int main(void) {
    printf("TraceLog: server: main() : E\n");

    int err = 0;
    int socket_fd = -1;

    // Create server socket file discriptor.
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

    // Remove old socket. If file exists, bind will be failed.
    remove(addr.sun_path);

    // Bind.
    err = bind(
            socket_fd, // Bind target socket.
            (struct sockaddr*) &addr, // Socket address.
            sizeof(struct sockaddr_un)); // Address size.
    if (err != 0) {
        on_error("bind failed.", socket_fd);
        return -1;
    }

    // Mark passive socket.
    err = listen(
            socket_fd, // Target socket.
            1); // Max connection count.
    if (err != 0) {
        on_error("listen failed.", socket_fd);
        return -1;
    }

    // Wait for connection and receive.
    while(1) {
        // Wait for client connection.
        int fd = accept(
                socket_fd, // Target socket.
                NULL, // Client address.
                NULL); // Client address size.
        if (fd == -1) {
            on_error("accept failed.", socket_fd);
            return -1;
        }

        // Receive.
        char buffer[256];
        int size = read(
                fd, // Accepted fd.
                buffer, // Receive buffer.
                sizeof(buffer) - 1); // Receive size. Last byte is null char.
        if (size == -1) {
            on_error("read failed.", socket_fd);
            return -1;
        }

        // Output received message.
        buffer[size] = '\0';
        printf("TraceLog: server: Received Msg = %s\n", buffer);

        // Close client fd.
        err = close(fd); // Closed client connection fd.
        if (err != 0) {
            on_error("close fd failed.", socket_fd);
            return -1;
        }

    } // while(true)

    // Close server socket.
    close(socket_fd);

    printf("## server main() : X\n");
    return 0;
}

