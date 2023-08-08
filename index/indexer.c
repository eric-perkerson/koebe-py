#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>


#define BUFFSIZE 4096



void copy (char from[], char *to, int len) {
  for (int i = 0; i < len; i++) {
    to[i] = from[i];
  } // for
} // copy

int parse (char *raw, int rawSize, long double *array, int arraySize, int fd) {
  int size = arraySize;
  int len = 0;
  int desiredNumFlag = 0;
  int bytesReadSinceNewline = 0;
  char *num = calloc(rawSize, sizeof(char));
  for (int i = 0; i < rawSize; i++) {
    //printf("Current character is %c\n", raw[i]);
    if (raw[i] == '\n') {
      num[len] = '\0';
      array[size] = strtold(num, NULL);
      //printf("Put in float: %.16Lf\n", array[size]);
      size++;
      len = 0;
      desiredNumFlag = 0;
      bytesReadSinceNewline = 0;
    } else if (raw[i] == ' ') {
      desiredNumFlag = 1;
    } else {
      if (desiredNumFlag == 1) {
	num[len] = raw[i];
	len++;
      } // if
    } // if
    bytesReadSinceNewline++;
  } // for
  lseek(fd, bytesReadSinceNewline * -1, SEEK_CUR);
  return size;
} // parse

int main (int argc, char *argv[]) {
  char *filePathPde;
  char *filePathTri;
  int opt;

  while((opt = getopt(argc, argv, "p:t:")) != -1) 
    { 
        switch(opt) 
        { 
            case 'p':
	      filePathPde = optarg;
	      break; 
            case 't': 
	      filePathTri = optarg;
	      break; 
            case '?': 
	      printf("unknown option: %c\n", optopt);
	      break; 
        } // switch
    } // while
  int pdeFd;
  int triFd;

  if ((pdeFd = open(filePathPde, O_RDONLY)) == -1) {
    perror("open");
    return EXIT_FAILURE;
  } // if



  char buf[BUFFSIZE];
  int n = -1;
  int header = 0;


  for (header = 0; buf[0] != '\n'; header++) {
    if (read(pdeFd, buf, 1) == -1) {
      perror("read");
      return EXIT_FAILURE;
    } // if
  } // for
  lseek(pdeFd, 0, SEEK_SET);
  if (read(pdeFd, buf, header) == -1) {
    perror("read");
    return EXIT_FAILURE;
  } // if
  buf[header - 1] = '\0';
  char length[header];
  copy(buf, length, header);
  printf("The length is %s\n", length);
  long double array[atoi(length)];
  int arraySize = 0;

  do {
    n = read(pdeFd, buf, BUFFSIZE);
    arraySize = parse(buf, n, array, arraySize, pdeFd);
  } while (n == BUFFSIZE);

  for (int i = 0; i < arraySize; i++) {
    printf("%.16Lf\n", array[i]);
  } // for
  close(pdeFd);
  if ((triFd = open(filePathTri, O_RDONLY)) == -1) {
    perror("open");
    return EXIT_FAILURE;
  } // if

  




  
  //  int pdeFd = open(filePath, 
} // main
