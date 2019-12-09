from hypothesis import settings, Verbosity

settings.register_profile("ci", max_examples=100)
settings.register_profile("dev", max_examples=7)
settings.register_profile("debug", max_examples=7, verbosity=Verbosity.verbose)
