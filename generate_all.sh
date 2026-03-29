for experiment in "velocity" "current" "radius" "side_length" "wavelength" "cross_section" "displacement" "market_cap"
do
    sbatch generate.sh $experiment
done