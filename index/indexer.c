#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>

#define BUFFSIZE 4096 // arbitrary buffer size for reading in data from the files


/*
 * Copies len characters from a character array into a different character array
 * from is the array being copied from
 * to is the array being copied to
 * len is the number of characters to be copied
 */
void copy (char from[], char *to, int len) {
  for (int i = 0; i < len; i++) {
    to[i] = from[i];
  } // for
} // copy

/*
 * This function parses the raw data read in from the .pde file and puts it into an array form
 * raw is the data being parsed
 * rawSize is the amount of data read in, in bytes
 * array is the array the data is being copied into
 * arraySize is what index array is on, ensuring the array is kept synchronized past buffer ends
 * fd is the file descriptor of the file being read
 */
int parse (char *raw, int rawSize, long double *array, int arraySize, int fd) {
  int size = arraySize; // keeps track of the arraySize to change later
  int len = 0; // length of the string being put into the array
  int desiredNumFlag = 0; // a flag so the parser skips over the first digit, as .pde is "vertex_number value"
  int bytesReadSinceNewline = 0; // number of bytes read since a new line is found, so the cursor in the file can be reset to the beginning of a line upon parse returning
  char num[rawSize]; // character array storing the string to be converted
  for (int i = 0; i < rawSize; i++) {
    if (raw[i] == '\n') {
      num[len] = '\0';
      array[size] = strtold(num, NULL);
      size++;
      len = 0;
      desiredNumFlag = 0;
      bytesReadSinceNewline = 0;
      /*
	If the character read is a newline, then the line is over, and the stored number is converted into
	a long double and stored in array
       */
    } else if (raw[i] == ' ') {
      desiredNumFlag = 1;
      // once the space is found, the remaining bytes need to be read in as the pde value
    } else {
      if (desiredNumFlag == 1) {
	num[len] = raw[i];
	len++;
      } // if
      // only lets data in if its the desired float we want to take in
    } // if
    bytesReadSinceNewline++;
  } // for
  lseek(fd, bytesReadSinceNewline * -1 + 1, SEEK_CUR); // puts the cursor on the beginning of the line so the next read call functions cleanly
  return size; // returns the current array size so parse is kept in sync
} // parse

/*
 * This is a parser for the .jos file, which contains the triangle topology for the mesh.
 * .jos is in the form:
 * number of verticies
 * neighboring_vertex_1 neighboring_vertex_2 neighboring_vertex_3 ...
 * ...
 * It turns the raw text data into a pointer to pointer int array.
 * The array has the form: array[vertex_number][vertex_neighbor_number]
 * with the special case at vertex_neighbor_number = 0, where it stores how many neighbors there are for the home vertex
 * raw is the data being parsed
 * rawSize is the amount of data read in, in bytes
 * array is the array the data is being copied into
 * arraySize is what index array is on, ensuring the array is kept synchronized past buffer ends
 * fd is the file descriptor of the file being read
 * length is the string containing the number of verticies
 */
int parseTri (char *raw, int rawSize, int **array, int arraySize, int fd, char *length) {
  int size = arraySize; // same as parse
  int bytesReadSinceNewline = 0; // same as parse
  int count = 1; // count keeps track of which neighbor has been read in
  int len = 0; // same as parse
  char str[strlen(length)]; // same as parse, except the length is capped at the length of the total number of verticies
  if (arraySize == 0) {
    array[size] = calloc(10, sizeof(int)); // initializes the first array being filled in for this parseTri call
  } // if
  for (int i = 0; i < rawSize; i++) {
    if (raw[i] == '\n') {
      str[len] = '\0';
      array[size][count] = atoi(str); // caps off the last number being stored and puts it into the array
      array[size][0] = count - 1;
      len = 0;
      count = 1;
      size++;
      bytesReadSinceNewline = 0;
      array[size] = calloc(10, sizeof(int));
      /*
	If the character is a newline, then all neighbors have been read in, thus the array being built is put into the pointer to pointer array
	and the next array is initialized
       */
    } else if (raw[i] == ' ') {
      str[len] = '\0';
      array[size][count] = atoi(str);
      len = 0;
      count++;
      /*
	Each space is the end of a neighbor, thus when encountered it caps off the string, and converts it into the array
       */
    } else {
      str[len] = raw[i];
      len++;
    } // if
    bytesReadSinceNewline++;
  } // for
  lseek(fd, bytesReadSinceNewline * -1 + 1, SEEK_CUR);
  return size;
} // parseTri

int main (int argc, char *argv[]) {
  char *filePathPde; // path for the pde file
  char *filePathTri; // path for the jos file
  int opt; // for option selecting

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
  int pdeFd; // the file descriptor for the pde file
  int triFd; // the file descriptor for the jos file

  if ((pdeFd = open(filePathPde, O_RDONLY)) == -1) {
    perror("open");
    return EXIT_FAILURE;
  } // if



  char buf[BUFFSIZE]; // buffer for reading in data
  int n = -1; // number of bytes read in
  int header = 0; // This will be used to read the header in, so we know the total amount of verticies


  for (header = 0; buf[0] != '\n'; header++) {
    if (read(pdeFd, buf, 1) == -1) {
      perror("read");
      return EXIT_FAILURE;
    } // if
  } // for
  // reads in characters 1 by 1 until the newline is found, storing where it was encountered
  lseek(pdeFd, 0, SEEK_SET);
  // returning to the beginning of the file
  if (read(pdeFd, buf, header) == -1) {
    perror("read");
    return EXIT_FAILURE;
  } // if
  // reads in the header
  buf[header - 1] = '\0';
  char length[header];
  copy(buf, length, header);
  // copies the header into a string
  printf("The length is %s\n", length);
  int vertexLen = atoi(length);
  // converts it into integer form
  
  long double pdeValues[vertexLen]; // array storing the solution function values
  int arraySize = 0; // keeps track of the array size

  do {
    n = read(pdeFd, buf, BUFFSIZE);
    arraySize = parse(buf, n, pdeValues, arraySize, pdeFd);
  } while (n == BUFFSIZE);
  // This reads in data and stores it into the array, until there is no more data to be read

  if (vertexLen != arraySize) {
    printf("VertexLen is %d, while arraySize is %d\n", vertexLen, arraySize);
  } // if
  // quick warning if the vertex numbers dont line up
  close(pdeFd);
  
  if ((triFd = open(filePathTri, O_RDONLY)) == -1) {
    perror("open");
    return EXIT_FAILURE;
  } // if

  for (header = 0; buf[0] != '\n'; header++) {
    if (read(triFd, buf, 1) == -1) {
      perror("read");
      return EXIT_FAILURE;
    } // if
  } // for
  // Skips past the header

  
  int **neighbor = calloc(vertexLen, sizeof(int*)); // creates the pointer to pointer array to be filled in using parseTri
  arraySize = 0; // resets the array size accordingly
  do {
    n = read(triFd, buf, BUFFSIZE);
    arraySize = parseTri(buf, n, neighbor, arraySize, triFd, length);
  } while (n == BUFFSIZE);
  // reads in data and stores it into the pointer to pointer array
  
  close(triFd);
  
  int indexes[vertexLen]; // index array, to be filled in with the indexes for each vertex
  int count = 0; // keeps track of sign changes
  long double change = 0; // variable storing the difference between home vertex and neighbor
  long double change2 = 0; // same as change, but stores the difference between home vertex and the NEXT neighbor

  
  for (int i = 0; i < vertexLen; i++) { // loops through all verticies
    if (neighbor[i][1] != -1) { // skips any boundry verticies
      for (int j = 1; j < neighbor[i][0]; j++) { // loops through every neighbor
	change = pdeValues[neighbor[i][j]] - pdeValues[i];
	change2 = pdeValues[neighbor[i][j + 1]] - pdeValues[i]; // change 1 and 2 are set to the differences
	count += (((change > 0) - (change < 0)) != ((change2 > 0) - (change2 < 0))); // increments count by 1 if there is a sign change between the two
      } // for
      change = pdeValues[neighbor[i][1]] - pdeValues[i]; // sets the difference for the loop around
      count += (((change > 0) - (change < 0)) != ((change2 > 0) - (change2 < 0)));
      indexes[i] = count; // fills in this vertex's spot in indexes with the number of sign changes
      count = 0;
    } else {
      indexes[i] = 2; // fills in any boundry points with 2 to ignore them
    } // if

    if (indexes[i] != 2) { // loops through each vertex and prints any with non-2 indicies
      printf("Found %d at %d\n", indexes[i], i);
    } // if
  } // for

  for (int i = 0; i < arraySize + 1; i++) {
    free(neighbor[i]);
  } // for
  free(neighbor);
 
} // main
