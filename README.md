# Hill-ADAM
Hill-ADAM is an optimizer that explores prescribed loss spaces with the goal of finding the global minimum. It does so in that it escapes local minima by alternating between minimization and maximization goals, allowing the optimizer to make its way over "hills" that the prescribed loss function may contain (these prevent ADAM from finding the global minimum). See Paper: http://arxiv.org/abs/2510.03613

# Applications
One such application of Hill-ADAM is image color correction. A source image's color palette, in general, is altered so that it matches that of a target image by training a model with a prescribed loss function (based on image distribution statistics) to learn the ideal RGB channel gains to apply to the source. Such color correction can ensure consistency between hundreds of images taken under different lighting conditions, which in turn can be used to increase accuracy while training predictive and generative AI models. This can also save manual labor and time in photo editing and graphics applications. 

Hill-ADAM is especially useful in instances where the target image is intended to guide rather than an exact reference of the color palette of the source image after color correction (ie. source-to-target color correction with additional constraints -- defined by user -- to palette). This is useful in photography and generative/predictive model training, where a certain mood is expected in a source image but the only available target images are not exactly ideal (ex. when an image needs to have cooler shades but target image is heavily blue-saturated).




