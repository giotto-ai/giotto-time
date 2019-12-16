from datetime import timedelta

from hypothesis import settings, Verbosity, HealthCheck

settings.register_profile(
    "ci",
    max_examples=100,
    suppress_health_check=(HealthCheck.too_slow,),
    deadline=timedelta(milliseconds=500),
)
settings.register_profile(
    "dev",
    max_examples=7,
    suppress_health_check=(HealthCheck.too_slow,),
    deadline=timedelta(milliseconds=500),
)
settings.register_profile("debug", max_examples=7, verbosity=Verbosity.verbose)
