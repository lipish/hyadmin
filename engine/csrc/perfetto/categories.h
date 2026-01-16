#ifndef TRACING_CATEGORIES_H
#define TRACING_CATEGORIES_H

#include "perfetto.h"

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("compute")
        .SetDescription("CPU compute"),
    perfetto::Category("schedule")
        .SetDescription("Thread scheduling"),
    perfetto::Category("taskqueue")
        .SetDescription("Task queue operations")
);

#endif  // TRACING_CATEGORIES_H