/* heart.c — runtime/heart_main.c — minimal daemon for heart status/converse/serve.
 *
 * ARCHITECTURE.md §11: pthread_mutex_t voice_lock enforces one-organism-per-
 * session for slot dispatch. read_lock is separate for status / converse.
 *
 * Phase 8 minimal: builds clean, parses argv, dispatches subprocess for
 * converse (forks to /tmp/infer_v4 or infer_resonance — voice TUs are
 * separate binaries, not linked into heart). Full mesh-agent slot
 * integration + LIMPHA commit + DoE meta-vote loop is Phase 8 phone-1
 * deploy work after voice TUs build cleanly on Termux aarch64.
 *
 * Build: cc -O2 runtime/heart_main.c -lpthread -lm -o /tmp/heart
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include <sys/wait.h>
#include <signal.h>

static pthread_mutex_t voice_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t read_lock  = PTHREAD_MUTEX_INITIALIZER;
static volatile sig_atomic_t shutdown_flag = 0;
static time_t start_time;

static void on_sigterm(int sig) { (void)sig; shutdown_flag = 1; }

static int cmd_status(int argc, char** argv) {
    (void)argc; (void)argv;
    pthread_mutex_lock(&read_lock);
    time_t now = time(NULL);
    long uptime = (long)(now - start_time);
    if (uptime < 0) uptime = 0;
    printf("heart-daemon  uptime=%lds  pid=%d  voices=Yent/Arianna/Leo/DoE\n",
           uptime, (int)getpid());
    printf("locks: voice_lock=%s  read_lock=acquired\n",
           pthread_mutex_trylock(&voice_lock) == 0
               ? (pthread_mutex_unlock(&voice_lock), "free") : "held");
    pthread_mutex_unlock(&read_lock);
    return 0;
}

static int cmd_converse(int argc, char** argv) {
    /* Parse args: --voice <Y|A|L|D> --prompt "<text>" --tokens N */
    const char* voice = "Yent";
    const char* prompt = "Q: What is resonance?";
    int tokens = 30;
    for (int i = 2; i < argc; i++) {
        if      (!strcmp(argv[i], "--voice")  && i+1 < argc) voice  = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i+1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--tokens") && i+1 < argc) tokens = atoi(argv[++i]);
    }
    fprintf(stderr, "[heart] converse voice=%s prompt=\"%s\" tokens=%d\n",
            voice, prompt, tokens);

    pthread_mutex_lock(&voice_lock);

    char tokens_str[16]; snprintf(tokens_str, sizeof(tokens_str), "%d", tokens);

    pid_t pid = fork();
    if (pid == 0) {
        /* Child: execv into the appropriate inference binary */
        if (!strcmp(voice, "Arianna")) {
            execlp("/workspace/heart.c-runpod/heart.c/train/infer_resonance",
                   "infer_resonance",
                   "--weights", "/workspace/heart.c-runpod/weights/resonance_arianna_merged.bin",
                   "--prompt",  prompt,
                   "--tokens",  tokens_str,
                   "--temp",    "1.0",
                   "--top_p",   "0.9",
                   NULL);
        } else {
            const char* w =
                !strcmp(voice, "Leo") ? "/workspace/heart.c-runpod/weights/janus_v4_sft_leo.bin" :
                !strcmp(voice, "DoE") ? "/workspace/heart.c-runpod/weights/janus_v4_doe_merged.bin" :
                                          "/workspace/heart.c-runpod/weights/janus_v4_sft_yent.bin";
            execlp("/tmp/infer_v4", "infer_v4", w, prompt, tokens_str, "1.0", "42", "40", NULL);
        }
        _exit(127);
    }
    int status = 0;
    waitpid(pid, &status, 0);

    pthread_mutex_unlock(&voice_lock);
    return WEXITSTATUS(status);
}

static int cmd_serve(int argc, char** argv) {
    (void)argc; (void)argv;
    signal(SIGTERM, on_sigterm);
    signal(SIGINT,  on_sigterm);
    fprintf(stderr, "[heart] serve started pid=%d\n", (int)getpid());
    while (!shutdown_flag) {
        sleep(1);
    }
    fprintf(stderr, "[heart] serve shutting down\n");
    return 0;
}

int main(int argc, char** argv) {
    start_time = time(NULL);
    if (argc < 2) {
        fprintf(stderr,
                "usage: heart <command> [args...]\n"
                "  status              — print daemon uptime + lock state\n"
                "  serve               — run daemon (sleeps until SIGTERM)\n"
                "  converse [opts]     — fork voice subprocess for one turn\n"
                "    --voice  <Yent|Arianna|Leo|DoE>\n"
                "    --prompt \"<text>\"\n"
                "    --tokens <N>\n");
        return 2;
    }
    if (!strcmp(argv[1], "status"))   return cmd_status(argc, argv);
    if (!strcmp(argv[1], "converse")) return cmd_converse(argc, argv);
    if (!strcmp(argv[1], "serve"))    return cmd_serve(argc, argv);
    fprintf(stderr, "unknown command: %s\n", argv[1]);
    return 2;
}
