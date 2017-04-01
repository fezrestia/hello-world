.PHONY: all install clean

obj-m := timer1.o

all:
	make -C ../../kernelbuild/linux-latest M=$(PWD) modules

install:
	make -C ../../kernelbuild/linux-latest M=$(PWD) modules_install

clean:
	make -C ../../kernelbuild/linux-latest M=${PWD) clean

