def transition_nwa(
        production_channel: tuple[tuple[str], tuple[str]],
        decay_channel: tuple[str],) -> tuple[str, tuple[str]]:
    final = [particle for particle in production_channel[1] if particle != 'alp']
    final += decay_channel
    return (production_channel[0], tuple(sorted(final)))