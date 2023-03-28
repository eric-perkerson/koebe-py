# makefile for Acute Software which also compiles Show Me
#
# Type "make" to compile Acute and Show Me.
# Type "make distclean" to delete all object and executable files.

# For more options please refer to Triangle's makefile
# http://www.cs.cmu.edu/~quake/triangle.html
# Here we give a sample compilation example for your convinience

# SRC is the directory in which the C source files are, and BIN is the
#   directory where you want to put the executable programs.  By default,
#   both are the current directory.

SRC = ./
BIN = ./

# CC should be set to the name of your favorite C compiler.

CC = cc
# for a Linux machine, the switches can be given as

# CSWITCHES = -O -DLINUX -I/usr/X11R6/include -L/usr/X11R6/lib

# RM should be set to the name of your favorite rm (file deletion program).

RM = /bin/rm

# The action starts here.

all: $(BIN)acute

$(BIN)acute: $(SRC)acute.c
	$(CC) $(CSWITCHES) -o $(BIN)acute $(SRC)acute.c -lm

distclean:
	$(RM) $(BIN)acute $(BIN)acute.o $(BIN)showme
