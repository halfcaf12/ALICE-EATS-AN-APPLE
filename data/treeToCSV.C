#include <stdio.h>
#include <unistd.h> // getcwd
#include <string.h> // string functionality
#include <stdlib.h> // heap
#include <dirent.h>
#include <iostream> // cout
#include <fstream>    // making a file
#include <sys/stat.h>  // make directory
#include <sys/types.h> // with permissions
/*
// we don't actually need to include these
#include "TFile.h"
#include "TTree.h"
#include "TList.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"
#include "TDirectoryFile.h"
#include "TString.h"
*/
using namespace std;

// load data from alice masterclass files at /eos/opendata/alice/masterclass/*.root

/* 
I have changed this to work off of the github as is, using 
dir/roots/ to get the root files 
storeCSV assumes existence of ./csvs/Clusters and ./csvs/RecTracks 
storeTree assumes existence of ./roots/Clusters and ./roots/RecTracks

scroll to bottom and find makeRoots, makeCSVS, and printLeaves flags
that you should change, I wasn't about to do a whole *args parsing
*/

/* get the current directory, globally*/
char DIRNAME[100];
char* ptr = getcwd(DIRNAME, sizeof(DIRNAME));
const char* ROOTDIR = "/roots"; // DIRECTORY TO SEARCH FOR .ROOT FILES
const char* CSVDIR  = "/csvs";  // DIRECTORY TO STORE CSV FILES

// print information about number of things iterated through 
typedef struct storeNums {
   int nFiles;   // number of ROOT files
   int nEvents;  // number of TDFiles
	int nTrees;   // number of trees
	int nEntries; // number of datapoints
} storeNums;

/* return all filepaths in directory. POINTER MUST BE FREED AFTERWARDS */
int void_strcmp(const void *str1, const void *str2) {
   return strcmp(*(const char **)(str1), *(const char **)(str2));
}

char** filenamesFromDir(const char* dir_path, const char* ending) {
   DIR *dir;
   struct dirent *entry;
   if ((dir = opendir(dir_path)) == NULL) { // couldn't open dir
      printf("Could not open %s",dir_path);
      return NULL;
   }
   // make char array pointer to all filepaths
   int storage = 5; int used = 0;
   char** filenames = (char **)calloc(sizeof(char *), storage); // initialize to zeros
   while ((entry = readdir(dir)) != NULL) {
      if (entry->d_name[0] == '.') continue; // no hidden files
      if (entry->d_type == DT_DIR) continue; // no directories
      if (ending) { // look for ending at the last part of the string that fits ending
         int length_to_end = strlen(entry->d_name) - strlen(ending);
         char* endstr = entry->d_name + length_to_end*sizeof(char);
         if (!strstr(endstr, ending)) continue;
      }
      if (used + 1 == storage) { // add 5 every time I guess
         storage += 5;
         filenames = (char **)realloc(filenames, storage * sizeof(char *));
      }
      filenames[used] = (char *)malloc(strlen(entry->d_name) + 1);
      strcpy(filenames[used], entry->d_name);
      used++;
   }
   filenames[used] = NULL;
   if (filenames[0] == NULL) {
      printf("directory %s is empty", dir_path);
      return NULL;
   } else { // sort filenames in ascending order
      qsort((void *)filenames, used, sizeof(filenames[0]), void_strcmp);
   }
   closedir(dir);
   return filenames;
}

// store tree into new ROOT file for viewing purposes
void storeTree(TTree* oldtree, const char* fname) {
   const char* tree_name = oldtree->GetName();
   // check that tree_name directory exists
   int len = strlen(DIRNAME)+strlen(ROOTDIR)+strlen(tree_name)+2;
   char treedir[len];
   snprintf(treedir, len, "%s%s/%s",DIRNAME,ROOTDIR,tree_name);
   DIR *dir;
   struct dirent *entry;
   if ((dir = opendir(treedir)) == NULL) { // need to make dir
      int status = mkdir(treedir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      if (status) {
         printf("\nABORTING storeTree... could not make directory %s",treedir);
         return;
      }
   }
   len = strlen(treedir)+strlen(fname)+strlen(tree_name)+8;
   char newfname[len];
   snprintf(newfname, len, "%s/%s_%s.root",treedir,fname,tree_name);
   oldtree->SetBranchStatus("*", 1);  // set all branches to be stored
   TFile newfile(newfname, "recreate");
   auto newtree = oldtree->CloneTree();
   newtree->Print();
   newfile.Write();
}

/* fname = name of file tree was found from */
void storeClustersCSV(TTree* tree, const char* fname, storeNums *NUMS) {
   ofstream myfile; // make output file
   myfile.open(fname); 
   // implement TTreeReader class
   TTreeReader reader(tree);
   
   // we know the columns...
   TTreeReaderValue<unsigned short int> f1s(reader, "fDetId");
   TTreeReaderValue<unsigned short int> f2s(reader, "fSubdetId");
   TTreeReaderValue<int> f3s(reader, "fLabel[3]");
   TTreeReaderValue<float> fXs(reader, "fV.fX");
   TTreeReaderValue<float> fYs(reader, "fV.fY");
   TTreeReaderValue<float> fZs(reader, "fV.fZ");
   myfile << "fDetId;fSubdetId;fLabel[3];fV.fX;fV.fY;fV.fZ\n";
   if (!myfile.is_open()) { 
      printf("we not gucci :(\n");
   }
   bool firstEntry = true;
   int i = 0;
   while (reader.Next()) {
      unsigned short int f1 = *f1s;
      unsigned short int f2 = *f2s;
      int f3 = *f3s;
      float fX = *fXs;
      float fY = *fYs;
      float fZ = *fZs;
      int len = snprintf(NULL,0,
      "%hu;%hu;%d;%f;%f;%f",
      f1,f2,f3,fX,fY,fZ)+1;
      char str[len];
      snprintf(str,len,
      "%hu;%hu;%d;%f;%f;%f",
      f1,f2,f3,fX,fY,fZ);
      myfile << str << "\n";
      i++;
   }
   if (reader.GetEntryStatus() != TTreeReader::kEntryBeyondEnd) {
      printf("WARNING! reader did not read all entries successfully");
   }
   printf(", %d; ",i);
   NUMS->nEntries += i;
   myfile.close();
}

void storeTracksCSV(TTree* tree, const char* fname, storeNums *NUMS) {
   ofstream myfile; // make output file
   myfile.open(fname);
   // implement TTreeReader class
   TTreeReader reader(tree);
   
   // we know the columns...
   TTreeReaderValue<float> PXs(reader, "fP.fX");
   TTreeReaderValue<float> PYs(reader, "fP.fY");
   TTreeReaderValue<float> PZs(reader, "fP.fZ");
   TTreeReaderValue<int> f1s(reader, "fLabel");
   TTreeReaderValue<int> f2s(reader, "fIndex");
   TTreeReaderValue<int> f3s(reader, "fStatus");
   TTreeReaderValue<int> f4s(reader, "fSign");
   TTreeReaderValue<float> VXs(reader, "fV.fX");
   TTreeReaderValue<float> VYs(reader, "fV.fY");
   TTreeReaderValue<float> VZs(reader, "fV.fZ");
   TTreeReaderValue<float> betas(reader, "fBeta");
   TTreeReaderValue<double> dcaxys(reader, "fDcaXY");
   TTreeReaderValue<double> dcazs(reader, "fDcaZ");
   TTreeReaderValue<double> fPVXs(reader, "fPVX");
   TTreeReaderValue<double> fPVYs(reader, "fPVY");
   TTreeReaderValue<double> fPVZs(reader, "fPVZ");
   myfile << "fLabel;fIndex;fStatus;fSign;fV.fX;fV.fY;fV.fZ;fP.fX;fP.fY;fP.fZ;fBeta;fDcaXY;fDcaZ;fPVX;fPVY;fPVZ\n";
   if (!myfile.is_open()) { 
      printf("we not gucci :(\n");
   }
   bool firstEntry = true;
   int i = 0;
   while (reader.Next()) {
      int f1 = *f1s;
      int f2 = *f2s;
      int f3 = *f3s;
      int f4 = *f4s;
      float VX = *VXs;
      float VY = *VYs;
      float VZ = *VZs;
      float pX = *PXs;
      float pY = *PYs;
      float pZ = *PZs;
      float beta = *betas;
      double dcaxy= *dcaxys;
      double dcaz = *dcazs;
      double PVX = *fPVXs;
      double PVY = *fPVYs;
      double PVZ = *fPVZs;
      int len = snprintf(NULL,0,
      "%d;%d;%d;%d;%f;%f;%f;%f;%f;%f;%f;%lf;%lf;%lf;%lf;%lf",
      f1,f2,f3,f4,VX,VY,VZ,pX,pY,pZ,beta,dcaxy,dcaz,PVX,PVY,PVZ)+1;
      char str[len];
      snprintf(str,len,
      "%d;%d;%d;%d;%f;%f;%f;%f;%f;%f;%f;%lf;%lf;%lf;%lf;%lf",
      f1,f2,f3,f4,VX,VY,VZ,pX,pY,pZ,beta,dcaxy,dcaz,PVX,PVY,PVZ);
      myfile << str << "\n";
      i++;
   }
   if (reader.GetEntryStatus() != TTreeReader::kEntryBeyondEnd) {
      printf("\nWARNING! reader did not read all entries successfully\n");
   }
   printf(", %d; ",i);
   NUMS->nEntries += i;
   myfile.close();
}

// useful for definining a new store___CSV to get names and types for tree leaves
void doPrintLeaves(TTree* tree, storeNums* NUMS) {
   TObjArray* leavesList = tree->GetListOfLeaves();
   int numLeaves = leavesList->GetEntries();
   NUMS->nEntries += numLeaves;
   printf(", %d\n",numLeaves);
   for (TObject* obj: *leavesList) {
      const char* typeName = obj->ClassName();
      const char* name = "";
      const char* type = "";
      if (!strcmp(typeName, "TLeafElement")) {
         TLeafElement* leaf = (TLeafElement *)obj;
         name = leaf->GetName();
         type = leaf->GetTypeName();
      } else if (!strcmp(typeName, "TLeafObject")) {
         TLeafObject* leaf = (TLeafObject *)obj;
         name = leaf->GetName();
         type = leaf->GetTypeName();
      } else if (!strcmp(typeName, "TLeaf")) {
         TLeaf* leaf = (TLeaf *)obj;
         name = leaf->GetName();
         type = leaf->GetTypeName();
      } else {
         printf("WARNING! unrecognized leaf type %s\n",typeName);
         continue;
      }
      printf("\t%s: %s\n",name,type);
   }
}

void processTree(TTree* tree, const char* fname, storeNums *NUMS, bool makeRoots, bool makeCSVs, bool printLeaves) {
   const char* tree_name = tree->GetName();
   printf("%s",tree_name);
   if (printLeaves) doPrintLeaves(tree, NUMS); 
   if (makeRoots) storeTree(tree, fname);
   if (makeCSVs) {
      // make sure directory is available
      int len = strlen(DIRNAME)+strlen(CSVDIR)+strlen(tree_name)+2;
      char csvpath[len];
      snprintf(csvpath, len, "%s%s/%s",DIRNAME,CSVDIR,tree_name);
      DIR *dir;
      struct dirent *entry;
      if ((dir = opendir(csvpath)) == NULL) { // need to make dir
         int status = mkdir(csvpath, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
         if (status) {
            printf("\nABORTING makeCSV... could not make directory %s\n",csvpath);
            return;
         }
      } // name path to csv file
      len = strlen(csvpath)+strlen(fname)+strlen(tree_name)+7;
      char newfname[len];
      snprintf(newfname, len, "%s/%s_%s.csv",csvpath,fname,tree_name);
      if (!strcmp(tree_name,"Clusters")) {
         storeClustersCSV(tree, newfname, NUMS);
      }
      if (!strcmp(tree_name, "RecTracks")) {
         storeTracksCSV(tree, newfname, NUMS);
      }
   }
}

void processEvent(TDirectoryFile* f, int fidx, storeNums *NUMS, bool makeRoots, bool makeCSVs, bool printLeaves) {
   // get Trees from event, such as clusters + rectracks
   TList *trees= f->GetListOfKeys();
   // make sure name can distinguish between same event, different masterclasses
   const char* fname = f->GetName();
   int fidx_len = snprintf(NULL,0,"%d",fidx);   // no ioa()?
   int len = strlen(fname) + 1 + fidx_len + 1;  //
   char storename[len];
   snprintf(storename, len, "%d_%s",fidx,fname);
   // iterate thru all trees in event
   for (int i=0; i < trees->GetEntries(); i++) {
      TObject* t = trees->At(i);
      const char* treename = t->GetName();
      TTree *tree = f->Get<TTree>(treename);
      processTree(tree, storename, NUMS, makeRoots, makeCSVs, printLeaves);
   }
   NUMS->nTrees += trees->GetEntries();
}

// ROOT calls the function that's named the same as the C file.
void treeToCSV() {
   // toggle storing singular .roots, storing csvs
   bool makeRoots   = false;
   bool makeCSVs    = false;
   bool printLeaves = false;
   bool onlyShowRootsFound = false; // if true, don't do anything
   // serach for filenames in DIRNAME that end with .root
   const char* ending = ".root";
   int len = strlen(DIRNAME) + strlen(ROOTDIR) + 1;
   char rootpath[len];
   snprintf(rootpath, len, "%s%s",DIRNAME,ROOTDIR);  // .root files located in roots/ folder
   char** filenames = filenamesFromDir(rootpath, ending);
   int fidx = 0;
   // for all ROOT files found
   storeNums *NUMS; NUMS->nFiles = 0; NUMS->nEvents = 0; NUMS->nTrees = 0; NUMS->nEntries = 0;
   while (char *filename = filenames[fidx]) {
      printf("%d - ",fidx);
      int len = strlen(rootpath)+strlen(filename)+2;
      char pathname[len];
      snprintf(pathname, len, "%s/%s", rootpath, filename);
      // open path to gets TFile with TDirs or TTrees
      TFile *fd = TFile::Open(pathname);
      TList *events= fd->GetListOfKeys();
      printf("processing %d entries in %s\n", events->GetEntries(),filename);
      if (!onlyShowRootsFound) {
      for(int i=0; i < events->GetEntries(); i++) {
         // events are either TDirectoryfile for masterclasses
         // or TTrees for ESD files
         TObject* t = events->At(i);
         const char* name = t->GetName();
         TObject* thing = fd->Get(name);
         const char* typeName = thing->ClassName();
         printf("\n%s: %s | ",typeName, name);
         if (!strcmp(typeName, "TDirectoryFile")) { 
            TDirectoryFile* event = fd->Get<TDirectoryFile>(name);
            processEvent(event, fidx, NUMS, makeRoots, makeCSVs, printLeaves);
            NUMS->nEvents++;
         }
         if (!strcmp(typeName, "TTree")) {
            TTree *tree = (TTree *)thing;
            processTree(tree, fd->GetName(), NUMS, makeRoots, makeCSVs, printLeaves);
            NUMS->nTrees++;
         }
      }
      }
      printf("freeing %s\n",filename);
      free(filename);  // was heaped
      fidx++;
   }
   printf("freeing %d filenames\n",fidx);
   free(filenames);  // was heaped
   printf("freed");
   NUMS->nFiles = fidx;
   printf("finished?");
   int numArray[4] = {NUMS->nFiles,NUMS->nEvents,NUMS->nTrees,NUMS->nEntries};
   const char *names[4]  = {"files","events","trees","entries"};
   const char *prev = "found";
   for (int i = 0; i < 4; i++) {
      int num = numArray[i];
      const char *name = names[i];
      if (!num) continue;
      int len = strlen(prev) + snprintf(NULL,0,"%d",num) + 4 + strlen(name);
      char newstring[len];
      snprintf(newstring,len,"%s %d %s,",prev,num,name);
      prev = newstring;
   }
   printf("%s",prev);
   return;
}