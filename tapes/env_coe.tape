package ersatz :: .versioner=disk .path="/home/hltcoe/rwicks/ersatz/code" {}

global {
	grid="sge"
	location="coe"
	cpuActionFlags=""
	cpuResourceFlags="-l 'mem_free=10G, ram_free=10G'"

	gpuActionFlags="-q gpu.q"
	gpuResourceFlags="-l 'gpu=1,h_rt=500:0:0'"

}
