export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
for i in  "pick" "place" "nav_to_obj"
do
find data/ -name ".habitat-resume-state*" -delete
python run.py \
    --config-name=rearrange/rl_skill.yaml \
    benchmark/rearrange=${i} \
    habitat_baselines.checkpoint_folder="data/checkpoints/models_tac_${i}"
mv data/checkpoints/models_tac_${i}/ckpt.19.pth data/models/${i}.pth
done