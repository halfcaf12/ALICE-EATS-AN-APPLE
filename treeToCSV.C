#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map> 
/*
#include "TFile.h"
#include "TTree.h"
#include "TList.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"
//#include "TriggerInfo.h"
#include "TDirectoryFile.h"
#include "TString.h"
*/
using namespace std;

/* 
load data from alice masterclass files at /eos/opendata/alice/masterclass/*.root

I have changed this to work off of the github as is, using 
dir/roots/ to get the root files 
storeCSV assumes existence of csvs/Clusters and csvs/RecTracks 
storeTree assumes existence of roots/Clusters and roots/RecTracks
*/
// const char* DIRNAME = "/Users/benkroul/Documents/CS/final_229/";
char DIRNAME[100];
if ( getcwd(DIRNAME, sizeof(DIRNAME)) == NULL ) {
   printf("getcwd() error");
   return 1;
}

/* return all filepaths in directory. POINTER MUST BE FREED AFTERWARDS */
char** filenamesFromDir(const char* dir_path, const char* ending) {
   DIR *dir;
   struct dirent *entry;
   if ((dir = opendir(dir_path)) == NULL) { // couldn't open dir
      printf("Could not open %s",dir_path);
      return NULL;
   }
   // make char array pointer to all filepaths
   int storage = 5; int used = 0;
   char** filenames = (char **)calloc(sizeof(char *), storage);
   while ((entry = readdir(dir)) != NULL) {
      if (entry->d_name[0] == '.') continue; // no hidden files
      if (entry->d_type == DT_DIR) continue; // no directories
      // filter for files with an ending
      if (ending && !strstr(entry->d_name, ending)) continue; 
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
   }
   closedir(dir);
   return filenames;
}

// store tree into new filename
void storeTree(const char* tree_name, TDirectoryFile* f, const int masterclass_index) {
   const char* fname = f->GetName();
   int len = strlen(DIRNAME)+strlen(fname)+17+2*strlen(tree_name);
   if (masterclass_index > 9) {
      len++;
   }
   char newfname[len];
   snprintf(newfname, len, "%sroots/%s/%d_%s_%s.root",DIRNAME,tree_name,masterclass_index,fname,tree_name);

   TTree* oldtree = f->Get<TTree>(tree_name);
   oldtree->SetBranchStatus("*", 1);
   TFile newfile(newfname, "recreate");
   auto newtree = oldtree->CloneTree();
   newtree->Print();
   newfile.Write();
}

void storeClustersCSV(const char* tree_name, TDirectoryFile* f, const int masterclass_index) {
   const char* fname = f->GetName();
   int len = strlen(DIRNAME)+strlen(fname)+15+2*strlen(tree_name);
   if (masterclass_index > 9) {
      len++;
   }
   char newfname[len];
   snprintf(newfname, len, "%scsvs/%s/%d_%s_%s.csv",DIRNAME,tree_name,masterclass_index,fname,tree_name);
   
   ofstream myfile; // make output file
   myfile.open(newfname);
   printf("%d_%s_%s.csv\n",masterclass_index,fname,tree_name);
   TTree *tree = f->Get<TTree>(tree_name);
   
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
      printf("we not guuci :(\n");
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
      printf("reader did not read all entries successfully");
   }
   printf("processed %d entries\n",i);
   myfile.close();
}

void storeTracksCSV(const char* tree_name, TDirectoryFile* f, const int masterclass_index) {
const char* fname = f->GetName();
   int len = strlen(DIRNAME)+strlen(fname)+15+2*strlen(tree_name);
   if (masterclass_index > 9) {
      len++;
   }
   char newfname[len];
   snprintf(newfname, len, "%scsvs/%s/%d_%s_%s.csv",DIRNAME,tree_name,masterclass_index,fname,tree_name);
   
   ofstream myfile; // make output file
   myfile.open(newfname);
   printf("%d_%s_%s.csv\n",masterclass_index,fname,tree_name);
   TTree *tree = f->Get<TTree>(tree_name);
   
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
      printf("we not guuci :(\n");
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
      printf("reader did not read all entries successfully");
   }
   printf("processed %d entries\n",i);
   myfile.close();
}

void doPrintLeaves(const char* tree_name, TDirectoryFile* f, const int masterclass_index) {
   TTree *tree = (TTree*)f->Get(tree_name);
   TObjArray* leavesList = tree->GetListOfLeaves();
   for (TObject* obj: *leavesList) {
      TLeaf* leaf = (TLeaf *)obj;
      const char* name = leaf->GetName();
      const char* type = leaf->GetTypeName();
      int len = strlen(name) + strlen(type) + 1 + 2;
      char printname[len];
      snprintf(printname, len, "%s, %s",name,type);
   }
}

// ROOT calls the function that's named the same as the C file.
void treeToCSV() {
   // toggle storing singular .roots, storing csvs
   bool makeRoots = false;
   bool makeCSVs  = true;
   bool printLeaves = true;
   // search for filenames in DIRNAME that end with .root
   const char* ending = ".root";
   int len = strlen(DIRNAME)+6;
   char rootpath[len];
   snprintf(rootpath, len, "%sroots",DIRNAME);
   char** filenames = filenamesFromDir(rootpath, ending);
   int file_idx = 0;
   // for all filenames found with .root ending
   while (char *filename = filenames[file_idx]) {
      int len = strlen(rootpath)+strlen(filename)+2;
      char pathname[len];
      snprintf(pathname, len, "%s/%s", rootpath, filename);
      // open path and get list of events (34 for index 1)
      TFile *fd = TFile::Open(pathname);
      TList *events= fd->GetListOfKeys();
      printf("found %d entries in %s\n", events->GetEntries(),filename);
      for(int i=0; i<events->GetEntries(); i++) {
         TDirectoryFile* f = (TDirectoryFile *)events->At(i);
         const char* fname = f->GetName();
         TDirectoryFile* event = (TDirectoryFile*)fd->Get(fname);

         if (printLeaves) {
            doPrintLeaves("Clusters", event, i);
            doPrintLeaves("RecTracks", event, i);
         }
         if (makeRoots) {
            storeTree("Clusters", event, i);
            storeTree("RecTracks", event, i);
         }
         if (makeCSVs) {
            storeClustersCSV("Clusters", event, i);
            storeTracksCSV("RecTracks", event, i);
         }
      }
      free(filename);
      file_idx++;
   }
   free(filenames);
   return;
}

// manually iterate through tree. might be the move for general csv 
// instead of stupid TTreeReader
/*
void storeCSV0(const char* tree_name, TDirectoryFile* f) {
   const char* fname = f->GetName();
   char newfname[43+strlen(fname)+strlen(tree_name)+5];
   strcpy(newfname, SAVE_PATH);
   strcat(newfname, fname);
   strcat(newfname, tree_name);
   strcat(newfname, ".txt\0");
   
   fstream myfile; // make output file
   myfile.open(newfname);

   TTree *tree = (TTree*)f->Get(tree_name);
   // tree->Scan(); // print content to screen
   float *addr; // create variables of the same type as the branches you want to access
   events->SetBranchAddress("branch_name",&addr); // for all the TTree branches you need this
   //Event *event = 0;  //event must be null or point to a valid object
                     //it must be initialized
   //tree.SetBranchAddress("event",&event);
   // loop over the tree
   for (int i=0;i<tree->GetEntries();i++){
      tree->GetEntry(i);
      //char* a = tree->Something;
      //myfile << a << "\n";
   }
   myfile.close();
}
*/

/* An attempt to make a general csv by generalizing the types of TTreeReaderValue */
// as it stands, each TTreeReaderValue is given a certain type and only exists
// in local scope, so you can't pass them around because of dereferencing
typedef struct readerWithType {
      const char* type;
      void* reader; // pointer to TTreeReaderValue
} readerWithType;

/*
readerWithType* makeTreeReaderFromType(const char* type, const char* name, TTreeReader reader) {
   // initialize a TTreeReaderValue() of the correct type
   void* ret = nullptr;
   if (!strcmp(type, "Int_t")) {
      TTreeReaderValue<Int_t>      ret(reader, name);
   } else if (!strcmp(type, "Float_t")) {
      TTreeReaderValue<Float_t>    ret(reader, name);
   } else if (!strcmp(type, "Double32_t")) {
      TTreeReaderValue<Double32_t> ret(reader, name);
   } else if (!strcmp(type, "UShort_t")) {
      TTreeReaderValue<UShort_t>   ret(reader, name);
   } else {
      printf("new type found in %s: %s",name,type);
   }
   readerWithType* newThing;
   newThing->type = type;
   newThing->reader = (void *)(&ret);
   return newThing;
}
*/

/*
void addReaderValueToFile(ofstream* fileptr, const readerWithType* thing) {
   const char* type = thing->type;
   char* str = NULL;
   if (!strcmp(type, "Int_t")) {
      // cast reader as pointer to TTreeReaderValue
      TTreeReaderValue<Int_t> reader = **(TTreeReaderValue<Int_t>*)thing->reader;
      // dereference reader to get TTreeReaderValue, then dereference it as iterator
      int val = (int)(*reader);
      int len = snprintf(NULL,0,"%d",val)+1;
      char str[len];
      snprintf(str,len,"%d",val);
   } else if (!strcmp(type, "Float_t")) {
      float val = (float)**(TTreeReaderValue<Float_t>*)thing->reader;
      int len = snprintf(NULL,0,"%f",val)+1;
      char str[len];
      snprintf(str,len,"%f",val);
   } else if (!strcmp(type, "Double32_t")) {
      double val = (double)**(TTreeReaderValue<Double_t>*)thing->reader;
      int len = snprintf(NULL,0,"%lf",val)+1;
      char str[len];
      snprintf(str,len,"%lf",val);
   } else if (!strcmp(type, "UShort_t")) {
      unsigned short int val = (unsigned short int)**(TTreeReaderValue<UShort_t>*)thing->reader;
      int len = snprintf(NULL,0,"%hu",val)+1;
      char str[len];
      snprintf(str,len,"%hu",val);
   }
   *fileptr << str;
}
*/

const char* valid_types[4] = {"Int_t","Float_t","Double32_t","UShort_t"};
// read tree and store in .txt file "csv"
/*
void storeCSV(const char* tree_name, TDirectoryFile* f, const int masterclass_index) {
   const char* fname = f->GetName();
   int len = strlen(DIRNAME)+strlen(fname)+15+2*strlen(tree_name);
   if (masterclass_index > 9) {
      len++;
   }
   char newfname[len];
   snprintf(newfname, len, "%scsvs/%s/%d_%s_%s.csv",DIRNAME,tree_name,masterclass_index,fname,tree_name);
   
   ofstream myfile; // make output file
   myfile.open(newfname);
   printf("%d_%s_%s.csv\n",masterclass_index,fname,tree_name);
   TTree *tree = f->Get<TTree>(tree_name);
   
   // implement TTreeReader class
   TTreeReader reader(tree);

   // dynamic TTreeReaderValue s, basically for columns
   vector<readerWithType*> column_readers;

   TObjArray* leavesList = tree->GetListOfLeaves();
   for (TObject* obj: *leavesList) {
      TLeaf* leaf = (TLeaf *)obj;
      const char* name = leaf->GetName();
      const char* type = leaf->GetTypeName();
      bool colIsValid = false;
      for (const char* typename: valid_types) {
         if (!strcmp(type, typename)) {
            colIsValid = true;
            break;
         }
      }
      if (!colIsValid) {
         continue;
      }
      //readerWithType* thing = makeTreeReaderFromType(type, name, reader);
      TTreeReaderValue ret(reader, name);
      readerWithType* thing;
      thing->type = type;
      thing->reader = ret;
      column_readers.push_back(thing);
      myfile << name;
   }
   myfile << "\n";
   if (!myfile.is_open()) { 
      printf("we not guuci :(\n");
   }

   bool firstEntry = true;
   int i = 0;
   while (reader.Next()) {
      for (int i = 0; i < column_readers.size(); i++) {
         readerWithType* col_reader = column_readers[i];
         if (col_reader->reader) {
            addReaderValueToFile(&myfile, col_reader);
         }
         myfile << str;
      }
      myfile << str << "\n";
      i++;
   }
   if (reader.GetEntryStatus() != TTreeReader::kEntryBeyondEnd) {
      printf("reader did not read all entries successfully");
   }
   printf("processed %d entries\n",i);
   myfile.close();
}
*/