for experiment in "velocity" "current" "radius" "side_length" "wavelength" "cross_section" "displacement" "market_cap"
do
    sbatch intervene_fix_traces.py --model Qwen2.5-32B --experiment $experiment --input_json 
done