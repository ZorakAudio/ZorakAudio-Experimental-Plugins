// SPDX-License-Identifier: Zlib
// Small real WDL/EEL semantic oracle. One process per fixture resets rand().
#include <fstream>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <string>
#include <sstream>
#define EEL_TARGET_PORTABLE 1
#define EELSCRIPT_NO_LICE 1
#define EELSCRIPT_NO_NET 1
#define EELSCRIPT_NO_FILE 1
#define EELSCRIPT_NO_MDCT 1
#define EELSCRIPT_NO_PREPROC 1
#include "WDL/eel2/eelscript.h"
static std::mutex m;
extern "C" void NSEEL_HOSTSTUB_EnterMutex(){m.lock();}
extern "C" void NSEEL_HOSTSTUB_LeaveMutex(){m.unlock();}
int main(int argc,char**argv){
 if(argc<2)return 2;
 std::ifstream in(argv[1]);std::ostringstream src;src<<in.rdbuf();
 NSEEL_init();eelScriptInst::init();eelScriptInst vm;const char*err=nullptr;
 auto code=vm.compile_code(src.str().c_str(),&err);
 if(!code){std::cerr<<(err?err:"compile error")<<'\n';return 1;}
 NSEEL_code_execute(code);
 std::cout<<std::setprecision(17);
 for(int i=2;i<argc;++i)std::cout<<argv[i]<<'='<<*NSEEL_VM_regvar(vm.m_vm,argv[i])<<'\n';
}
