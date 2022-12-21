#include <rdycore.h>

static const char *help_str = "rdycore - a standalone driver for RDycore\n"
"usage: rdycore [options] <filename>\n";

int main(int argc, char *argv[]) {
  PetscCall(RDyInit(argc, argv, help_str));
  PetscCall(RDyFinalize());

  return 0;
}
