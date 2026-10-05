// Plugin manifest, collected into the webapp by `invoke dev.install`.
export default {
  apiVersion: 1,
  blocks: {
    "insitu-xrd": () => import("./XRDInsituBlock.vue"),
    "insitu-nmr": () => import("./NMRInsituBlock.vue"),
    "insitu-uvvis": () => import("./UVVisInsituBlock.vue"),
  },
};
