#define _GNU_SOURCE
#include <dlfcn.h>

#include <stdio.h>
#include <time.h>
#include <string.h>
 
static FILE *fp_trace;
int depth = 0;
static int MAX_DEPTH = 10;

extern "C" {
 
void __attribute__ ((constructor)) trace_begin (void) {
    fp_trace = fopen("trace.out", "w");
    depth = 0;
}
 
void __attribute__ ((destructor)) trace_end (void) {
    if(fp_trace != NULL) {
        fclose(fp_trace);
    }
}
 
void __cyg_profile_func_enter (void *func,  void *caller) {
    if (fp_trace != NULL) {
        if (depth < MAX_DEPTH) {
            Dl_info info;
            if (dladdr(func, &info)) {
                if (info.dli_sname) {
                    if (!strstr(info.dli_sname, "St3")) {
                        fprintf (fp_trace, "%i %p %p [%s] %s\n",
                            depth,
                            func,
                            caller,
                            info.dli_fname ? info.dli_fname : "?",
                            info.dli_sname ? info.dli_sname : "?");
                        fflush(fp_trace);
                    }
                }
            }
        }
        depth++;
    }
}
 
void __cyg_profile_func_exit (void *func, void *caller) {
    if(fp_trace != NULL) {
        //fprintf(fp_trace, "x %p %p %lu\n", func, caller, time(NULL));
        depth--;
    }
    
}

}
