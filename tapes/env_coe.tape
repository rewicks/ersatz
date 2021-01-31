package ersatz :: .versioner=disk .path="/home/hltcoe/rwicks/ersatz/code" {}

global {
	grid="sge"
	location="coe"
	cpuActionFlags=""
	cpuResourceFlags="-l 'h_rt=100:0:0'"

	gpuActionFlags="-q gpu.q -M rwicks@jhu.edu -m base"
	gpuResourceFlags="-l 'gpu=1,h_rt=500:0:0'"

}
