package ersatz :: .versioner=disk .path="/home/hltcoe/rwicks/ersatz/code/ersatz" {}

global {
	grid="sge"
	location="coe"
	cpuActionFlags=""
	cpuResourceFlags="-l 'h_rt=100:0:0'"

	gpuActionFlags="-q gpu.q@@v100 -M rwicks@jhu.edu -m base"
	gpuResourceFlags="-l 'gpu=1,h_rt=500:0:0'"

}
